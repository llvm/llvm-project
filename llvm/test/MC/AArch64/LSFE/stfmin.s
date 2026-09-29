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
// STFMIN
//------------------------------------------------------------------------------

stfmin h0, [x2]
// CHECK-INST: stfmin h0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c20d05f <unknown>

stfmin h2, [sp]
// CHECK-INST: stfmin h2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c22d3ff <unknown>

stfmin s0, [x2]
// CHECK-INST: stfmin s0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc20d05f <unknown>

stfmin s2, [sp]
// CHECK-INST: stfmin s2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc22d3ff <unknown>

stfmin d0, [x2]
// CHECK-INST: stfmin d0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc20d05f <unknown>

stfmin d2, [sp]
// CHECK-INST: stfmin d2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc22d3ff <unknown>

// --stfminl

stfminl h0, [x2]
// CHECK-INST: stfminl h0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c60d05f <unknown>

stfminl h2, [sp]
// CHECK-INST: stfminl h2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c62d3ff <unknown>

stfminl s0, [x2]
// CHECK-INST: stfminl s0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc60d05f <unknown>

stfminl s2, [sp]
// CHECK-INST: stfminl s2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc62d3ff <unknown>

stfminl d0, [x2]
// CHECK-INST: stfminl d0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc60d05f <unknown>

stfminl d2, [sp]
// CHECK-INST: stfminl d2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc62d3ff <unknown>

//------------------------------------------------------------------------------
// STBFMIN
//------------------------------------------------------------------------------

stbfmin h0, [x2]
// CHECK-INST: stbfmin h0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c20d05f <unknown>

stbfmin h2, [sp]
// CHECK-INST: stbfmin h2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c22d3ff <unknown>

// -- stbfminl

stbfminl h0, [x2]
// CHECK-INST: stbfminl h0, [x2]
// CHECK-ENCODING: [0x5f,0xd0,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c60d05f <unknown>

stbfminl h2, [sp]
// CHECK-INST: stbfminl h2, [sp]
// CHECK-ENCODING: [0xff,0xd3,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c62d3ff <unknown>