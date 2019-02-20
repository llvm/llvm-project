// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+ssbs  < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+v8.5a < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-ssbs  < %s 2>&1 | FileCheck %s --check-prefix=NOSPECID

mrs x2, SSBS

// CHECK:         mrs x2, {{ssbs|SSBS}} // encoding: [0xc2,0x42,0x3b,0xd5]
// NOSPECID:      error: expected readable system register
// NOSPECID-NEXT: mrs x2, {{ssbs|SSBS}}

msr SSBS, x3
msr SSBS, #1

// CHECK:         msr {{ssbs|SSBS}}, x3 // encoding: [0xc3,0x42,0x1b,0xd5]
// CHECK:         msr {{ssbs|SSBS}}, #1 // encoding: [0x3f,0x41,0x03,0xd5]
// NOSPECID:      error: expected writable system register or pstate
// NOSPECID-NEXT: msr {{ssbs|SSBS}}, x3
// NOSPECID:      error: expected writable system register or pstate
// NOSPECID-NEXT: msr {{ssbs|SSBS}}, #1
