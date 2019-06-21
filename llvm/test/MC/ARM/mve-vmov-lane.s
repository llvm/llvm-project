// RUN: not llvm-mc -triple=thumbv8m.main   -mattr=+fp-armv8 -show-encoding < %s 2>%t | FileCheck %s --check-prefix=V80M
// RUN: FileCheck %s < %t --check-prefix=V80M-ERROR
// RUN:     llvm-mc -triple=thumbv8.1m.main -mattr=+fp-armv8 -show-encoding < %s 2>%t
// RUN:     llvm-mc -triple=thumbv8.1m.main -mattr=+mve      -show-encoding < %s 2>%t

// v8.1M added the Q register syntax for this instruction. The v8.1M spec does
// not list the D register syntax as valid, but we accept it as an extension to
// make porting code from v8.0M to v8.1M easier.

vmov.32 r0, d1[0]
// V80M: vmov.32 r0, d1[0]               @ encoding: [0x11,0xee,0x10,0x0b]
// V81M: vmov.32 r0, d1[0]               @ encoding: [0x11,0xee,0x10,0x0b]

vmov.32 r0, q0[2]
// V80M-ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction requires: armv8.1m.main with FP or MVE
// V81M: vmov.32 r0, q0[2]               @ encoding: [0x11,0xee,0x10,0x0b]
