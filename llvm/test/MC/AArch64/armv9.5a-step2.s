// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x0, MDSTEPOP_EL1
// CHECK: mrs x0, MDSTEPOP_EL1                  // encoding: [0x40,0x05,0x30,0xd5]

msr MDSTEPOP_EL1, x0
// CHECK: msr MDSTEPOP_EL1, x0                  // encoding: [0x40,0x05,0x10,0xd5]
