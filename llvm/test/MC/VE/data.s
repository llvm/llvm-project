# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -s - | \
# RUN:     FileCheck %s --check-prefix=OBJ

.data
a:
.2byte 0xff55, 0x88aa
.4byte 0xff5588aa, 0xdeadbeaf
.8byte 0xff5588aadeadbeaf, 0xdeadbeafdeadbeaf
.byte 0xff, 0x55, 0x88
.short 0xff55, 0x88aa
.word 0xff5588aa, 0xdeadbeaf
.int 0xff5588aa, 0xdeadbeaf
.long 0xff5588aadeadbeaf, 0xdeadbeafdeadbeaf
.quad 0xff5588aadeadbeaf, 0xdeadbeafdeadbeaf
.llong 0xff5588aadeadbeaf, 0xdeadbeafdeadbeaf

# CHECK:      .2byte  65365
# CHECK-NEXT: .2byte  34986
# CHECK-NEXT: .4byte  4283795626
# CHECK-NEXT: .4byte  3735928495
# CHECK-NEXT: .8byte  -47981953555775825
# CHECK-NEXT: .8byte  -2401053363754123601
# CHECK-NEXT: .byte   255
# CHECK-NEXT: .byte   85
# CHECK-NEXT: .byte   136
# CHECK-NEXT: .2byte  65365
# CHECK-NEXT: .2byte  34986
# CHECK-NEXT: .4byte  4283795626
# CHECK-NEXT: .4byte  3735928495
# CHECK-NEXT: .4byte  4283795626
# CHECK-NEXT: .4byte  3735928495
# CHECK-NEXT: .8byte  -47981953555775825
# CHECK-NEXT: .8byte  -2401053363754123601
# CHECK-NEXT: .8byte  -47981953555775825
# CHECK-NEXT: .8byte  -2401053363754123601
# CHECK-NEXT: .8byte  -47981953555775825
# CHECK-NEXT: .8byte  -2401053363754123601

# OBJ:      Contents of section .data:
# OBJ-NEXT: 0000 55ffaa88 aa8855ff afbeadde afbeadde
# OBJ-NEXT: 0010 aa8855ff afbeadde afbeadde ff558855
# OBJ-NEXT: 0020 ffaa88aa 8855ffaf beaddeaa 8855ffaf
# OBJ-NEXT: 0030 beaddeaf beaddeaa 8855ffaf beaddeaf
# OBJ-NEXT: 0040 beaddeaf beaddeaa 8855ffaf beaddeaf
# OBJ-NEXT: 0050 beaddeaf beaddeaa 8855ffaf beaddeaf
# OBJ-NEXT: 0060 beadde
