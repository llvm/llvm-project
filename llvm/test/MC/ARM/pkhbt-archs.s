@ RUN: llvm-mc -triple thumbv7 %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-VALID
@ RUN: llvm-mc -triple thumbv8 %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-VALID
@ RUN: llvm-mc -triple thumbv7em %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-VALID
@ RUN: llvm-mc -triple thumbv6t2 %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-VALID

@ RUN: not llvm-mc -triple thumbv6 %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
@ RUN: not llvm-mc -triple thumbv7m %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
@ RUN: not llvm-mc -triple thumbv8m.main %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
@ RUN: not llvm-mc -triple thumbv8m.base %s -o - -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID

         @ Instruction is "v6T2, v7" in ARMARM-AR, "v7em" in ARMARM-M. So it's
         @ valid on everything v6t2 upwards, except v7m. Also apparently not on
         @ v8m (going by present behaviour).
         pkhbt r1, r2, r3, lsl #24

@ CHECK-VALID: pkhbt r1, r2, r3, lsl #24     @ encoding: [0xc2,0xea,0x03,0x61]
@ CHECK-INVALID: error:
