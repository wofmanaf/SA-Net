import torch
import torch.nn.functional as F


class GroupAct(object):
    def __init__(self, model, target_layer="layer4.2", groups=64):
        super(GroupAct, self).__init__()
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = model.cuda().eval()
        self.groups = groups
        self.activations = dict()

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        if torch.cuda.is_available():
            input = input.cuda()

        # predication on raw input
        logit = self.model(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value'].data

        activations = activations.chunk(self.groups, 1)
        scores = []
        with torch.no_grad():
            for act in activations:
                act = torch.mean(act, dim=1, keepdim=True)
                act = F.interpolate(act, size=(h, w), mode='bilinear', align_corners=False)
                act = (act - act.min()) / (act.max() - act.min())
                output = self.model(input * act)
                output = F.softmax(output, dim=-1)
                score = output[0][predicted_class]
                scores.append(score.item())

        return scores

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)