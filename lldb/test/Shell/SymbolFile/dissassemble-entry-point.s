# REQUIRES: lld, arm

# RUN: llvm-mc -triple=thumbv7-eabi %s -filetype=obj -o %t.o
# RUN: ld.lld %t.o -o %t --image-base=0x8000 --section-start=.text=0x8074 -e 0x8075 -s
# RUN: %lldb -x -b -o 'dis -s 0x8074 -e 0x8080' -- %t | FileCheck %s
# CHECK:      {{.*}}[0x8074] <+0>: movs   r0, #0x2a
# CHECK-NEXT: {{.*}}[0x8076] <+2>: movs   r7, #0x1
# CHECK-NEXT: {{.*}}[0x8078] <+4>: svc    #0x0

_start:
    movs r0, #0x2a
    movs r7, #0x1
    svc #0x0
