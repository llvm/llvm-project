// RUN: llvm-mc -triple=aarch64 -show-encoding  < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj  < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj  < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// read

mrs x3, FPMR
// CHECK-INST: mrs x3, FPMR
// CHECK-ENCODING: [0x43,0x44,0x3b,0xd5]
// CHECK-UNKNOWN: d53b4443   mrs   x3, FPMR


mrs x3, ID_AA64FPFR0_EL1
// CHECK-INST: mrs x3, ID_AA64FPFR0_EL1
// CHECK-ENCODING: [0xe3,0x04,0x38,0xd5]
// CHECK-UNKNOWN: d53804e3   mrs   x3, ID_AA64FPFR0_EL1

// --------------------------------------------------------------------------//
// write

msr FPMR, x3
// CHECK-INST: msr FPMR, x3
// CHECK-ENCODING: [0x43,0x44,0x1b,0xd5]
// CHECK-UNKNOWN: d51b4443   msr   FPMR, x3