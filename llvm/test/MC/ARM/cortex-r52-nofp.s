@ RUN: llvm-mc -triple armv8-none-eabi -mcpu=cortex-r52 -mattr=+nosimd+nofp.dp %s -o - | FileCheck %s -check-prefix=CHECK-NO-FP
@ RUN: llvm-mc -triple armv8-none-eabi -mcpu=cortex-r52 %s -o - | FileCheck %s -check-prefix=CHECK-FP

.text
vadd.f32 s0, s1, s2
@ CHECK-NO-FP: vadd.f32 s0, s1, s2
@ CHECK-FP: vadd.f32 s0, s1, s2
@ CHECK-NOT-NO-FP: error: instruction requires: VPF2
@ CHECK-NOT-FP: error: instruction requires: VPF2
