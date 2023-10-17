@ RUN: llvm-mc -triple armv7-elf -filetype obj -o - %s | llvm-readelf -S -s - | FileCheck %s

@ CHECK:   [Nr] Name              Type            Address  Off    Size   ES Flg Lk Inf Al
@ CHECK-NEXT:   [ 0]                   NULL            00000000 000000 000000 00      0   0  0
@ CHECK-NEXT:   [ 1] .strtab           STRTAB          00000000 {{.*}} {{.*}} 00      0   0  1
@ CHECK-NEXT:   [ 2] .text             PROGBITS        00000000 {{.*}} 00000d 00  AX  0   0  4
@ CHECK-NEXT:   [ 3] .arm_aligned      PROGBITS        00000000 {{.*}} 000005 00  AX  0   0  4
@ CHECK-NEXT:   [ 4] .thumb_aligned    PROGBITS        00000000 {{.*}} 000002 00  AX  0   0  2

@ CHECK:      Num:    Value  Size Type    Bind   Vis      Ndx Name
@ CHECK-NEXT:   0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND
@ CHECK-NEXT:   1: 00000001     0 FUNC    LOCAL  DEFAULT    2 aligned_thumb
@ CHECK-NEXT:   2: 00000000     0 NOTYPE  LOCAL  DEFAULT    2 $t.0
@ CHECK-NEXT:   3: 00000004     0 FUNC    LOCAL  DEFAULT    2 thumb_to_arm
@ CHECK-NEXT:   4: 00000004     0 NOTYPE  LOCAL  DEFAULT    2 $a.1
@ CHECK-NEXT:   5: 00000008     0 NOTYPE  LOCAL  DEFAULT    2 $d.2
@ CHECK-NEXT:   6: 0000000b     0 FUNC    LOCAL  DEFAULT    2 unaligned_arm_to_thumb
@ CHECK-NEXT:   7: 0000000a     0 NOTYPE  LOCAL  DEFAULT    2 $t.3

.thumb

.type aligned_thumb,%function
aligned_thumb:
    nop

@ Above function has size 2 (at offset 0)
@ Expect alignment of +2 (to offset 4)
.arm

.type thumb_to_arm,%function
thumb_to_arm:
    nop

.byte 0

@ Above function has size 5 (at offset 4)
@ Expect alignment of +1 (to offset 10)
.thumb
.type unaligned_arm_to_thumb,%function
unaligned_arm_to_thumb:
    nop

.byte 0

@ Above section has size 13 (at offset 34)
@ Expect alignment of +3 (to offset 44)
.section .arm_aligned, "ax"
.arm

.type arm_aligned_section,%function
arm_aligned_section:
    nop

.byte 0

@ Above section has size 5 (at offset 44)
@ Expect alignment of +1 (to offset 4a)
.section .thumb_aligned, "ax"
.thumb

.type thumb_aligned_section,%function
thumb_aligned_section:
    nop
