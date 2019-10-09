        .text
        .byte   0
sizeless:
sizeful:
        .byte   0
        .byte   0
sizeend:
        .size   sizeful, sizeend - sizeful
