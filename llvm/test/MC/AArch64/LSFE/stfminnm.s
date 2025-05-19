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
// STFMINNM
//------------------------------------------------------------------------------

stfminnm h0, [x2]
// CHECK-INST: stfminnm h0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c20f05f <unknown>

stfminnm h2, [sp]
// CHECK-INST: stfminnm h2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c22f3ff <unknown>

stfminnm s0, [x2]
// CHECK-INST: stfminnm s0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc20f05f <unknown>

stfminnm s2, [sp]
// CHECK-INST: stfminnm s2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc22f3ff <unknown>

stfminnm d0, [x2]
// CHECK-INST: stfminnm d0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc20f05f <unknown>

stfminnm d2, [sp]
// CHECK-INST: stfminnm d2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc22f3ff <unknown>

// -- stfminnml

stfminnml h0, [x2]
// CHECK-INST: stfminnml h0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c60f05f <unknown>

stfminnml h2, [sp]
// CHECK-INST: stfminnml h2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c62f3ff <unknown>

stfminnml s0, [x2]
// CHECK-INST: stfminnml s0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc60f05f <unknown>

stfminnml s2, [sp]
// CHECK-INST: stfminnml s2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc62f3ff <unknown>

stfminnml d0, [x2]
// CHECK-INST: stfminnml d0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc60f05f <unknown>

stfminnml d2, [sp]
// CHECK-INST: stfminnml d2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc62f3ff <unknown>

//------------------------------------------------------------------------------
// STBFMINNM
//------------------------------------------------------------------------------

stbfminnm h0, [x2]
// CHECK-INST: stbfminnm h0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c20f05f <unknown>

stbfminnm h2, [sp]
// CHECK-INST: stbfminnm h2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c22f3ff <unknown>

// -- stbfminnml

stbfminnml h0, [x2]
// CHECK-INST: stbfminnml h0, [x2]
// CHECK-ENCODING: [0x5f,0xf0,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c60f05f <unknown>

stbfminnml h2, [sp]
// CHECK-INST: stbfminnml h2, [sp]
// CHECK-ENCODING: [0xff,0xf3,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c62f3ff <unknown>