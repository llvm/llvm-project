@ RUN: not llvm-mc -mcpu=cortex-a8 -triple armv7-unknown-unknown -show-encoding -mattr=-neon < %s 2>&1 | FileCheck %s --check-prefix=VFP --check-prefix=CHECK
@ RUN: not llvm-mc -mcpu=cortex-a8 -triple thumbv7-unknown-unknown -show-encoding -mattr=-neon < %s 2>&1 | FileCheck %s --check-prefix=VFP --check-prefix=CHECK
@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-unknown-unknown -show-encoding -mattr=+neon < %s 2>&1 | FileCheck %s --check-prefix=NEON --check-prefix=CHECK
@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-unknown-unknown -show-encoding -mattr=+neon < %s 2>&1 | FileCheck %s --check-prefix=NEON --check-prefix=CHECK

@ The 32-bit variants of the NEON scalar move instructions are also available
@ to any core with VFPv2

@ CHECK-DAG: vmov.32 d13[0], r6 @ encoding:
@ CHECK-DAG: vmov.32 d17[1], r9 @ encoding:
vmov.32 d13[0], r6
vmov.32 d17[1], r9

@ VFP-DAG: error: instruction requires: NEON
@ VFP-DAG: error: instruction requires: NEON
@ NEON-DAG: vmov.8  d22[5], r2 @ encoding:
@ NEON-DAG: vmov.16 d3[2], r4 @ encoding:
vmov.8 d22[5], r2
vmov.16 d3[2], r4

@ CHECK-DAG: vmov.32 r6, d13[0] @ encoding:
@ CHECK-DAG: vmov.32 r9, d17[1] @ encoding:
vmov.32 r6, d13[0]
vmov.32 r9, d17[1]

@ VFP-DAG: error: instruction requires: NEON
@ VFP-DAG: error: instruction requires: NEON
@ NEON-DAG: vmov.s8 r2, d22[5] @ encoding:
@ NEON-DAG: vmov.u16        r4, d3[2] @ encoding:
vmov.s8 r2, d22[5]
vmov.u16 r4, d3[2]

