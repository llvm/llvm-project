// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lsfe < %s \
// RUN:        | llvm-objdump -d --mattr=+lsfe - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lsfe < %s \
// RUN:        | llvm-objdump -d --mattr=-lsfe - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+lsfe -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// STFADD
//------------------------------------------------------------------------------

stfadd h0, [x2]
// CHECK-INST: stfadd h0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c20805f <unknown>

stfadd h2, [sp]
// CHECK-INST: stfadd h2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c2283ff <unknown>

stfadd s0, [x2]
// CHECK-INST: stfadd s0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc20805f <unknown>

stfadd s2, [sp]
// CHECK-INST: stfadd s2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc2283ff <unknown>

stfadd d0, [x2]
// CHECK-INST: stfadd d0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc20805f <unknown>

stfadd d2, [sp]
// CHECK-INST: stfadd d2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc2283ff <unknown>

// -- stfaddl

stfaddl h0, [x2]
// CHECK-INST: stfaddl h0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c60805f <unknown>

stfaddl h2, [sp]
// CHECK-INST: stfaddl h2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c6283ff <unknown>

stfaddl s0, [x2]
// CHECK-INST: stfaddl s0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc60805f <unknown>

stfaddl s2, [sp]
// CHECK-INST: stfaddl s2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc6283ff <unknown>

stfaddl d0, [x2]
// CHECK-INST: stfaddl d0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc60805f <unknown>

stfaddl d2, [sp]
// CHECK-INST: stfaddl d2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc6283ff <unknown>

//------------------------------------------------------------------------------
// STBFADD
//------------------------------------------------------------------------------

stbfadd h0, [x2]
// CHECK-INST: stbfadd h0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c20805f <unknown>

stbfadd h2, [sp]
// CHECK-INST: stbfadd h2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c2283ff <unknown>

// -- stbfaddl

stbfaddl h0, [x2]
// CHECK-INST: stbfaddl h0, [x2]
// CHECK-ENCODING: [0x5f,0x80,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c60805f <unknown>

stbfaddl h2, [sp]
// CHECK-INST: stbfaddl h2, [sp]
// CHECK-ENCODING: [0xff,0x83,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c6283ff <unknown>