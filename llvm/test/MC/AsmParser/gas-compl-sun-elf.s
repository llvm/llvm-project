@ RUN: llvm-mc -filetype=obj -triple arm-linux-gnu %s -o - | llvm-readelf --sections | FileCheck %s

@ CHECK: .f1              PROGBITS        00000000 000034 000000 00   A  0   0  1
.section ".f1", #alloc

@ CHECK: .f2              PROGBITS        00000000 000034 000000 00   W  0   0  1
.section ".f2", #write

@ CHECK: .f3              PROGBITS        00000000 000034 000000 00   A  0   0  1
.section ".f3", "a"

@ CHECK: .f4              PROGBITS        00000000 000034 000000 00   W  0   0  1
.section ".f4", "w"
