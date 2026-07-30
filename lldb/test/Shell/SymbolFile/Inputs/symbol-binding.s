        .text
        .byte   0
sizeless:
sizeful:
        .byte   0
        .byte   0
sizeend:
        .size   sizeful, sizeend - sizeful
        .byte   0
case1_local:
case1_global:
        .globl  case1_global
        .byte   0
case2_local:
case2_weak:
        .weak   case2_weak
        .byte   0
case3_weak:
        .weak   case3_weak
case3_global:
        .globl  case3_global
        .byte   0
