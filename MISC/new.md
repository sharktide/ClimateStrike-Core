#### 10/13/2025

Today was the first day I started working on this

I am using FIRMS' data in csv form to try to make my first model to predict wildfires.

No luck today: The model learned a shortcut that if temperature > 300 K, 100% else 0%. Nothing else matters.
I'm going to have to introduce penalties and suppressors and boosters into the architecture and try again. But that is for tommorow.

#### 10/14/2025


I finally got FireNet working! Actually it was a lot harder than that.

FireNet is an Artificial Neural Network with the Tabular Classification Pipeline. The model would just REFUSE to obay common sense, so I had to program in lambda functions into the architecture itself. I am going to apply the sigmoid activation to the final fully connected (dense) layer, but then have **that** layer be the ``raw_output``. Then I am going to define some lambda layers with the sole purpose of boosting or supressing the model's confidence based on real life conditions, and multiple those with the raw outtput to get our ANN.

Long story short: It worked and the model was making reasonable predictions.

But not good enough as I wanted them to be; after all, if this is going to be implmented in REAL LIFE it cannot make these mistakes.

So then I designed a Tabular Regression model (FireTrustNet) that, when trained on some different data (I had no problems here since I am already randomly downsampling FIRMS), essentially decides how much we should "trust" the base model (FireNet). Using Tabular Regression, some complex math to guide how the model should behave, and a custom activation function (yes really!), I got it to output a sigmoid in the range of 0.5 - 1.5. This is then used to alter FireNet's predictions

#### 10/15/2025

Today I made on a rudimentary gradio app and some trust-modulation plots to see how well the model behaved. It took me two hours but it was worth it. I now see that FireNet and FireTrustNet preform reasonably well together and am going to move on to predicting Fluvial Floods (FV-FloodNet & FV-FloodTrustNet)

FV-FloodNet just wouldn't work. For a while it was just predicting either 0% or 100%, and I think the dataset is either horrible (after all It was synthetic) or my inference test script just isn't working.

#### 10/16/2025

Well I decided to train FV-FloodNet on only a small subset of the dataset: 1000 of yes flood and 1000 of no flood.
After some testing, I realized that the penalties and boosters are causing vanishing gradients, bu when I removed it, the accuracy still wasn't good enough. I then tried putting the boosts/supression back but using soft logits as well as moving the sigmoid activation but no luck. I spent hours debugging, but in the end I found that the most effective way was to move the sigmoid to the adjusted output and use an add layer instead of multiplication for the penalties and boosters. It now works quite well! 

I also made FV-FloodTrustNet, and that also works, but I made a mistake along the way. I trained FV-FloodNet with the class names in snake_case while FV-FloodNet was traind on Pascal Case. Even though this isn't a breaking issue, it will be a pain to deal with.

I added this all to the gradio app, as well as added options to convert metrics and a "Feature Definitions" tab so people can understand better what information the model needs.

I am going to spend the rest of the month improving my open-soure ML library [tensorflowtools](https://github.com/sharktide/tftools), contributing to CPython (yes im a python contributor) and working on [reSructuredPython](https://restructuredpython.rf.gd): a superset of python that I made

#### 10/17/2025

Today I focused a bit more on PV-FloodNet. I got a working version that is preforming okay, but I noticed something troubling: the rainfall gradient is vanishing. 

I even did a test and found out my rainfal gradient was less than 0.000001!

I now am a victim of the vanishing gradient problem!

I tried the following

- Processing rainfall seperatley
- Making residual connections
- Experimenting with sigmoid
- Adding modulation clipping
- Removing Normalization (yes i may be crazy)
- Ragequitting

NOTHING WORKED

Then I looked at the training logs

I found guess what - 100% validation accuracy! I was confused until I realized the horrible realities of Deep Neural Networks and Synthetic Dataset Generation

- A validation accuracy of 100% is rarely good: it usually indicates a problem with your dataset
- I generated my dataset with NumPy
- No matter how much ranodm loc i did it didn't matter
- It didn't matter because I was not explictly having noise in the rainfall column
- The model learned that rainfall wasn't important because of the dataset
- Nothing i did with the architecture mattered because the dataset was fundametally horrible.

Anyway I fixed everything and now the gradient is alive and I even did a plot and some inference tests and its looking good! I put it in MISC/misc_code/plot.repy

I'll continue work tomorrow

#### 10/18/2025

Its preforming so well I don't think I need a TrustNet! However I think I'll make one anyway; tradition after all!

#### 10/19/2025

Ran some tests

#### 10/20/2025

Added the third model to the gradio app. Will test tomorrow

#### 10/21/2025

I wasted my entire day because I messed up the lambda expression. Fixed it and uploaded the bew space to huggingface

#### 10/22/2025

Today I put myself into turbo mode and did the followinbg:

- Pushed the logs
- Made FlashoodNet and FlashFloodTrustNet
- Tested them
- Added them to the gradio app
- Added then to the HF space
- Staged, Comitted, and Pushed Everything
  
#### 10/23/2025

Made QuakeNet, QuakeTrustNet and added them to the gradio app

#### 10/24/2025

Made HurricaneNet and HurricaneTrustNet and added them to the gradio app,

#### 10/25/2025

Started working on the app. It'll be a dotnet maui blazor/mudblazor hybrid

#### 10/26/2025

Continued working on the app.

#### 10/27/2025

Created TornadoNet and TornadoTrustNet and added them to the gradio app.
