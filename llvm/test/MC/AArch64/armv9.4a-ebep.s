// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x2, PM
msr PM, x3
msr PM, #1

// CHECK:       mrs x2, {{pm|PM}} // encoding: [0x22,0x43,0x38,0xd5]
// CHECK:       msr {{pm|PM}}, x3 // encoding: [0x23,0x43,0x18,0xd5]
// CHECK:       msr {{pm|PM}}, #1 // encoding: [0x1f,0x43,0x01,0xd5]
