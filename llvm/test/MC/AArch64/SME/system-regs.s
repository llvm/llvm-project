// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// read

mrs x3, ID_AA64SMFR0_EL1
// CHECK-INST: mrs x3, ID_AA64SMFR0_EL1
// CHECK-ENCODING: [0xa3,0x04,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: a3 04 38 d5   mrs   x3, S3_0_C0_C4_5

mrs x3, SMCR_EL1
// CHECK-INST: mrs x3, SMCR_EL1
// CHECK-ENCODING: [0xc3,0x12,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: c3 12 38 d5   mrs   x3, S3_0_C1_C2_6

mrs x3, SMCR_EL2
// CHECK-INST: mrs x3, SMCR_EL2
// CHECK-ENCODING: [0xc3,0x12,0x3c,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: c3 12 3c d5   mrs   x3, S3_4_C1_C2_6

mrs x3, SMCR_EL3
// CHECK-INST: mrs x3, SMCR_EL3
// CHECK-ENCODING: [0xc3,0x12,0x3e,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: c3 12 3e d5   mrs   x3, S3_6_C1_C2_6

mrs x3, SMCR_EL12
// CHECK-INST: mrs x3, SMCR_EL12
// CHECK-ENCODING: [0xc3,0x12,0x3d,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: c3 12 3d d5   mrs   x3, S3_5_C1_C2_6

mrs x3, SVCR
// CHECK-INST: mrs x3, SVCR
// CHECK-ENCODING: [0x43,0x42,0x3b,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 43 42 3b d5   mrs   x3, S3_3_C4_C2_2

mrs x3, SMPRI_EL1
// CHECK-INST: mrs x3, SMPRI_EL1
// CHECK-ENCODING: [0x83,0x12,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 83 12 38 d5   mrs   x3, S3_0_C1_C2_4

mrs x3, SMPRIMAP_EL2
// CHECK-INST: mrs x3, SMPRIMAP_EL2
// CHECK-ENCODING: [0xa3,0x12,0x3c,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: a3 12 3c d5   mrs   x3, S3_4_C1_C2_5

mrs x3, SMIDR_EL1
// CHECK-INST: mrs x3, SMIDR_EL1
// CHECK-ENCODING: [0xc3,0x00,0x39,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: c3 00 39 d5   mrs   x3, S3_1_C0_C0_6

mrs x3, TPIDR2_EL0
// CHECK-INST: mrs x3, TPIDR2_EL0
// CHECK-ENCODING: [0xa3,0xd0,0x3b,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: a3 d0 3b d5   mrs   x3, S3_3_C13_C0_5

// --------------------------------------------------------------------------//
// write

msr SMCR_EL1, x3
// CHECK-INST: msr SMCR_EL1, x3
// CHECK-ENCODING: [0xc3,0x12,0x18,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: c3 12 18 d5   msr   S3_0_C1_C2_6, x3

msr SMCR_EL2, x3
// CHECK-INST: msr SMCR_EL2, x3
// CHECK-ENCODING: [0xc3,0x12,0x1c,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: c3 12 1c d5   msr   S3_4_C1_C2_6, x3

msr SMCR_EL3, x3
// CHECK-INST: msr SMCR_EL3, x3
// CHECK-ENCODING: [0xc3,0x12,0x1e,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: c3 12 1e d5   msr   S3_6_C1_C2_6, x3

msr SMCR_EL12, x3
// CHECK-INST: msr SMCR_EL12, x3
// CHECK-ENCODING: [0xc3,0x12,0x1d,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: c3 12 1d d5   msr   S3_5_C1_C2_6, x3

msr SVCR, x3
// CHECK-INST: msr SVCR, x3
// CHECK-ENCODING: [0x43,0x42,0x1b,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 43 42 1b d5   msr   S3_3_C4_C2_2, x3

msr SMPRI_EL1, x3
// CHECK-INST: msr SMPRI_EL1, x3
// CHECK-ENCODING: [0x83,0x12,0x18,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 83 12 18 d5   msr   S3_0_C1_C2_4, x3

msr SMPRIMAP_EL2, x3
// CHECK-INST: msr SMPRIMAP_EL2, x3
// CHECK-ENCODING: [0xa3,0x12,0x1c,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: a3 12 1c d5   msr   S3_4_C1_C2_5, x3

msr SVCRSM, #0
// CHECK-INST: smstop sm
// CHECK-ENCODING: [0x7f,0x42,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 7f 42 03 d5   msr   S0_3_C4_C2_3, xzr

msr SVCRSM, #1
// CHECK-INST: smstart
// CHECK-ENCODING: [0x7f,0x43,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 7f 43 03 d5   msr   S0_3_C4_C3_3, xzr

msr SVCRZA, #0
// CHECK-INST: smstop za
// CHECK-ENCODING: [0x7f,0x44,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 7f 44 03 d5   msr   S0_3_C4_C4_3, xzr

msr SVCRZA, #1
// CHECK-INST: smstart za
// CHECK-ENCODING: [0x7f,0x45,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 7f 45 03 d5   msr   S0_3_C4_C5_3, xzr

msr SVCRSMZA, #0
// CHECK-INST: smstop
// CHECK-ENCODING: [0x7f,0x46,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 7f 46 03 d5   msr   S0_3_C4_C6_3, xzr

msr SVCRSMZA, #1
// CHECK-INST: smstart
// CHECK-ENCODING: [0x7f,0x47,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 7f 47 03 d5   msr   S0_3_C4_C7_3, xzr

msr TPIDR2_EL0, x3
// CHECK-INST: msr TPIDR2_EL0, x3
// CHECK-ENCODING: [0xa3,0xd0,0x1b,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: a3 d0 1b d5   msr   S3_3_C13_C0_5, x3
