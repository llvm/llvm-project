@ RUN: llvm-mc -triple thumbv8.1m.main-arm-none-eabi -arm-implicit-it=always -mattr=+pacbti < %s -show-encoding | FileCheck %s

autgeq r0, r1, r2
pacgeq r0, r1, r2
bxauteq r0, r1, r2
@ CHECK: ittt eq
@ CHECK: autgeq r0, r1, r2
@ CHECK: pacgeq r0, r1, r2
@ CHECK: bxauteq r0, r1, r2
