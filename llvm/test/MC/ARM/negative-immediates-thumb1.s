# RUN: llvm-mc -triple thumbv7 -mcpu=cortex-m0 %s -show-encoding | FileCheck %s
# RUN: not llvm-mc -triple thumbv7 -mcpu=cortex-m0 %s -show-encoding -mattr=+no-neg-immediates 2>&1 | FileCheck %s -check-prefix=CHECK-DISABLED

.thumb

	ADDs r1, r0, #0xFFFFFFF9
# CHECK: subs r1, r0, #7
# CHECK-DISABLED: error: instruction requires: NegativeImmediates
	ADDs r0, #0xFFFFFF01
# CHECK: subs r0, #255
# CHECK-DISABLED: error: instruction requires: NegativeImmediates

	SUBs r0, #0xFFFFFF01
# CHECK: adds r0, #255
# CHECK-DISABLED: error: instruction requires: NegativeImmediates

	SUBs r1, r0, #0xFFFFFFF9
# CHECK: adds r1, r0, #7
# CHECK-DISABLED: error: instruction requires: NegativeImmediates
