// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// read

mrs x3, ID_AA64SMFR0_EL1
// CHECK-INST: mrs x3, ID_AA64SMFR0_EL1
// CHECK-ENCODING: [0xa3,0x04,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53804a3   mrs   x3, S3_0_C0_C4_5

mrs x3, SMCR_EL1
// CHECK-INST: mrs x3, SMCR_EL1
// CHECK-ENCODING: [0xc3,0x12,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53812c3   mrs   x3, S3_0_C1_C2_6

mrs x3, SMCR_EL2
// CHECK-INST: mrs x3, SMCR_EL2
// CHECK-ENCODING: [0xc3,0x12,0x3c,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53c12c3   mrs   x3, S3_4_C1_C2_6

mrs x3, SMCR_EL3
// CHECK-INST: mrs x3, SMCR_EL3
// CHECK-ENCODING: [0xc3,0x12,0x3e,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53e12c3   mrs   x3, S3_6_C1_C2_6

mrs x3, SMCR_EL12
// CHECK-INST: mrs x3, SMCR_EL12
// CHECK-ENCODING: [0xc3,0x12,0x3d,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53d12c3   mrs   x3, S3_5_C1_C2_6

mrs x3, SVCR
// CHECK-INST: mrs x3, SVCR
// CHECK-ENCODING: [0x43,0x42,0x3b,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53b4243   mrs   x3, S3_3_C4_C2_2

mrs x3, SMPRI_EL1
// CHECK-INST: mrs x3, SMPRI_EL1
// CHECK-ENCODING: [0x83,0x12,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d5381283   mrs   x3, S3_0_C1_C2_4

mrs x3, SMPRIMAP_EL2
// CHECK-INST: mrs x3, SMPRIMAP_EL2
// CHECK-ENCODING: [0xa3,0x12,0x3c,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53c12a3   mrs   x3, S3_4_C1_C2_5

mrs x3, SMIDR_EL1
// CHECK-INST: mrs x3, SMIDR_EL1
// CHECK-ENCODING: [0xc3,0x00,0x39,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53900c3   mrs   x3, S3_1_C0_C0_6

mrs x3, TPIDR2_EL0
// CHECK-INST: mrs x3, TPIDR2_EL0
// CHECK-ENCODING: [0xa3,0xd0,0x3b,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53bd0a3   mrs   x3, S3_3_C13_C0_5

// --------------------------------------------------------------------------//
// write

msr SMCR_EL1, x3
// CHECK-INST: msr SMCR_EL1, x3
// CHECK-ENCODING: [0xc3,0x12,0x18,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51812c3   msr   S3_0_C1_C2_6, x3

msr SMCR_EL2, x3
// CHECK-INST: msr SMCR_EL2, x3
// CHECK-ENCODING: [0xc3,0x12,0x1c,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51c12c3   msr   S3_4_C1_C2_6, x3

msr SMCR_EL3, x3
// CHECK-INST: msr SMCR_EL3, x3
// CHECK-ENCODING: [0xc3,0x12,0x1e,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51e12c3   msr   S3_6_C1_C2_6, x3

msr SMCR_EL12, x3
// CHECK-INST: msr SMCR_EL12, x3
// CHECK-ENCODING: [0xc3,0x12,0x1d,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51d12c3   msr   S3_5_C1_C2_6, x3

msr SVCR, x3
// CHECK-INST: msr SVCR, x3
// CHECK-ENCODING: [0x43,0x42,0x1b,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51b4243   msr   S3_3_C4_C2_2, x3

msr SMPRI_EL1, x3
// CHECK-INST: msr SMPRI_EL1, x3
// CHECK-ENCODING: [0x83,0x12,0x18,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d5181283   msr   S3_0_C1_C2_4, x3

msr SMPRIMAP_EL2, x3
// CHECK-INST: msr SMPRIMAP_EL2, x3
// CHECK-ENCODING: [0xa3,0x12,0x1c,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51c12a3   msr   S3_4_C1_C2_5, x3

msr SVCRSM, #0
// CHECK-INST: smstop sm
// CHECK-ENCODING: [0x7f,0x42,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d503427f   smstop sm

msr SVCRSM, #1
// CHECK-INST: smstart
// CHECK-ENCODING: [0x7f,0x43,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d503437f   smstart

msr SVCRZA, #0
// CHECK-INST: smstop za
// CHECK-ENCODING: [0x7f,0x44,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d503447f   smstop za

msr SVCRZA, #1
// CHECK-INST: smstart za
// CHECK-ENCODING: [0x7f,0x45,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d503457f   smstart za

msr SVCRSMZA, #0
// CHECK-INST: smstop
// CHECK-ENCODING: [0x7f,0x46,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d503467f   smstop

msr SVCRSMZA, #1
// CHECK-INST: smstart
// CHECK-ENCODING: [0x7f,0x47,0x03,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d503477f   smstart

msr TPIDR2_EL0, x3
// CHECK-INST: msr TPIDR2_EL0, x3
// CHECK-ENCODING: [0xa3,0xd0,0x1b,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51bd0a3   msr   S3_3_C13_C0_5, x3
