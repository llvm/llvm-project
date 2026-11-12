// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+nmi   < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v8.8a < %s | FileCheck %s

mrs x2, ALLINT
msr ALLINT, x3
msr ALLINT, #1
mrs x7, ICC_NMIAR1_EL1

// CHECK:       mrs x2, {{allint|ALLINT}} // encoding: [0x02,0x43,0x38,0xd5]
// CHECK:       msr {{allint|ALLINT}}, x3 // encoding: [0x03,0x43,0x18,0xd5]
// CHECK:       msr {{allint|ALLINT}}, #1 // encoding: [0x1f,0x41,0x01,0xd5]
// CHECK:       mrs x7, {{icc_nmiar1_el1|ICC_NMIAR1_EL1}} // encoding: [0xa7,0xc9,0x38,0xd5]
