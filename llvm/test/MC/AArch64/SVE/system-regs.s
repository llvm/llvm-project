// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

mrs x3, ID_AA64ZFR0_EL1
// CHECK-INST: mrs x3, ID_AA64ZFR0_EL1
// CHECK-ENCODING: [0x83,0x04,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 83 04 38 d5   mrs   x3, S3_0_C0_C4_4

mrs x3, ZCR_EL1
// CHECK-INST: mrs x3, ZCR_EL1
// CHECK-ENCODING: [0x03,0x12,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 03 12 38 d5   mrs   x3, S3_0_C1_C2_0

mrs x3, ZCR_EL2
// CHECK-INST: mrs x3, ZCR_EL2
// CHECK-ENCODING: [0x03,0x12,0x3c,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 03 12 3c d5   mrs   x3, S3_4_C1_C2_0

mrs x3, ZCR_EL3
// CHECK-INST: mrs x3, ZCR_EL3
// CHECK-ENCODING: [0x03,0x12,0x3e,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 03 12 3e d5   mrs   x3, S3_6_C1_C2_0

mrs x3, ZCR_EL12
// CHECK-INST: mrs x3, ZCR_EL12
// CHECK-ENCODING: [0x03,0x12,0x3d,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 03 12 3d d5   mrs   x3, S3_5_C1_C2_0

msr ZCR_EL1, x3
// CHECK-INST: msr ZCR_EL1, x3
// CHECK-ENCODING: [0x03,0x12,0x18,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 03 12 18 d5   msr   S3_0_C1_C2_0, x3

msr ZCR_EL2, x3
// CHECK-INST: msr ZCR_EL2, x3
// CHECK-ENCODING: [0x03,0x12,0x1c,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 03 12 1c d5   msr   S3_4_C1_C2_0, x3

msr ZCR_EL3, x3
// CHECK-INST: msr ZCR_EL3, x3
// CHECK-ENCODING: [0x03,0x12,0x1e,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 03 12 1e d5   msr   S3_6_C1_C2_0, x3

msr ZCR_EL12, x3
// CHECK-INST: msr ZCR_EL12, x3
// CHECK-ENCODING: [0x03,0x12,0x1d,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 03 12 1d d5   msr   S3_5_C1_C2_0, x3
