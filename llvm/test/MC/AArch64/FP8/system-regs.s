// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fpmr < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fpmr < %s \
// RUN:        | llvm-objdump -d --mattr=+fpmr - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fpmr < %s \
// RUN:        | llvm-objdump --mattr=-fpmr -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// read

mrs x3, FPMR
// CHECK-INST: mrs x3, FPMR
// CHECK-ENCODING: [0x43,0x44,0x3b,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53b4443   mrs   x3, S3_3_C4_C4_2

mrs x3, ID_AA64FPFR0_EL1
// CHECK-INST: mrs x3, ID_AA64FPFR0_EL1
// CHECK-ENCODING: [0xe3,0x04,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: d53804e3   mrs   x3, S3_0_C0_C4_7

// --------------------------------------------------------------------------//
// write

msr FPMR, x3
// CHECK-INST: msr FPMR, x3
// CHECK-ENCODING: [0x43,0x44,0x1b,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: d51b4443   msr   S3_3_C4_C4_2, x3
