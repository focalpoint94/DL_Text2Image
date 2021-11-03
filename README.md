# DL_Text2Image
딥러닝 강좌의 Text2Image Project Repo입니다.

Text Input에 대해 256*256*3 Size Image를 생성하는 Project입니다.

# Model Description
Text Input을 Gan의 Generator에 통과시켜 64X64X3 size image를 생성합니다.

Generator에서 생성된 64X64X3 size image를 ESDR Network에 통과시켜 256X256X3 size image를 생성합니다.

<img src = "https://github.com/focalpoint94/DL_Text2Image/blob/main/data/Model.JPG?raw=true" width="100%" height="100%">

# Results
## Text Input
```
"a black bird with oily black feathers and rounded black beak."
"a medium sized black bird, with a white belly, and webbed feet."
"this is a white bird with black webbed feet and a black beak."
"a small dully colored bird that has a grey head and nape, an oatmeal colored breast, belly and yellow and oatmeal-grey colored wings and tail."
"this bird has a yellow throat, breast and belly, with a black band at the throat, and black crown, wings and tail."
"this small bird is soft green all over, with a light eyering, short wings and moderate tail."
"small bird with crown and throat is yellow, outer and inner rectrices are grey, beak is small, black and pointed."
"a brown bird with striped wings, a large head, a long pointy beak, a long tail, and narrow legs."
```

## Generator Output (64X64X3)

<img src = "https://github.com/focalpoint94/DL_Text2Image/blob/main/data/64sized.png?raw=true" width="50%" height="50%">


## ESDR Output (256X256X3)

<img src = "https://github.com/focalpoint94/DL_Text2Image/blob/main/data/256sized.png?raw=true" width="50%" height="50%">


