from options.test_options1 import TestOptions
from models import create_model
import os
from Setting import project_path
opt = TestOptions().parse()

opt.model = 'AdATTACK'
opt.net = 'search'

AdA = create_model(opt)
AdA.load_path = os.path.join(project_path,'checkpoints/%s/model.pth'%opt.model)
AdA.setup(opt)
AdA.eval()
