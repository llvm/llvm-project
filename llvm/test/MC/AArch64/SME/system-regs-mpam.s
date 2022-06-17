// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme,+mpam < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+mpam < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme,+mpam < %s \
// RUN:        | llvm-objdump -d --mattr=+sme,+mpam - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme,+mpam < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme,+mpam < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme,+mpam < %s \
// RUN:        | llvm-objdump -d --mattr=+mpam - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// read

mrs x3, MPAMSM_EL1
// CHECK-INST: mrs x3, MPAMSM_EL1
// CHECK-ENCODING: [0x63,0xa5,0x38,0xd5]
// CHECK-ERROR: expected readable system register
// CHECK-UNKNOWN: 63 a5 38 d5   mrs   x3, S3_0_C10_C5_3

// --------------------------------------------------------------------------//
// write

msr MPAMSM_EL1, x3
// CHECK-INST: msr MPAMSM_EL1, x3
// CHECK-ENCODING: [0x63,0xa5,0x18,0xd5]
// CHECK-ERROR: expected writable system register or pstate
// CHECK-UNKNOWN: 63 a5 18 d5   msr   S3_0_C10_C5_3, x3
