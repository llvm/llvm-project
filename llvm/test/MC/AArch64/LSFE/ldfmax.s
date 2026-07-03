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
// LDFMAX
//------------------------------------------------------------------------------

ldfmax h0, h1, [x2]
// CHECK-INST: ldfmax h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c204041 <unknown>

ldfmax h2, h3, [sp]
// CHECK-INST: ldfmax h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c2243e3 <unknown>

ldfmax s0, s1, [x2]
// CHECK-INST: ldfmax s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc204041 <unknown>

ldfmax s2, s3, [sp]
// CHECK-INST: ldfmax s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc2243e3 <unknown>

ldfmax d0, d1, [x2]
// CHECK-INST: ldfmax d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc204041 <unknown>

ldfmax d2, d3, [sp]
// CHECK-INST: ldfmax d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc2243e3 <unknown>

// -- ldfmaxa

ldfmaxa h0, h1, [x2]
// CHECK-INST: ldfmaxa h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xa0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca04041 <unknown>

ldfmaxa h2, h3, [sp]
// CHECK-INST: ldfmaxa h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xa2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca243e3 <unknown>

ldfmaxa s0, s1, [x2]
// CHECK-INST: ldfmaxa s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xa0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca04041 <unknown>

ldfmaxa s2, s3, [sp]
// CHECK-INST: ldfmaxa s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xa2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca243e3 <unknown>

ldfmaxa d0, d1, [x2]
// CHECK-INST: ldfmaxa d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xa0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca04041 <unknown>

ldfmaxa d2, d3, [sp]
// CHECK-INST: ldfmaxa d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xa2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca243e3 <unknown>

// -- ldfmaxal

ldfmaxal h0, h1, [x2]
// CHECK-INST: ldfmaxal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xe0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce04041 <unknown>

ldfmaxal h2, h3, [sp]
// CHECK-INST: ldfmaxal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xe2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce243e3 <unknown>

ldfmaxal s0, s1, [x2]
// CHECK-INST: ldfmaxal s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xe0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce04041 <unknown>

ldfmaxal s2, s3, [sp]
// CHECK-INST: ldfmaxal s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xe2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce243e3 <unknown>

ldfmaxal d0, d1, [x2]
// CHECK-INST: ldfmaxal d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xe0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce04041 <unknown>

ldfmaxal d2, d3, [sp]
// CHECK-INST: ldfmaxal d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xe2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce243e3 <unknown>

// -- ldfmaxl

ldfmaxl h0, h1, [x2]
// CHECK-INST: ldfmaxl h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c604041 <unknown>

ldfmaxl h2, h3, [sp]
// CHECK-INST: ldfmaxl h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c6243e3 <unknown>

ldfmaxl s0, s1, [x2]
// CHECK-INST: ldfmaxl s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc604041 <unknown>

ldfmaxl s2, s3, [sp]
// CHECK-INST: ldfmaxl s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc6243e3 <unknown>

ldfmaxl d0, d1, [x2]
// CHECK-INST: ldfmaxl d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc604041 <unknown>

ldfmaxl d2, d3, [sp]
// CHECK-INST: ldfmaxl d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc6243e3 <unknown>

//------------------------------------------------------------------------------
// LDBFMAX
//------------------------------------------------------------------------------

ldbfmax h0, h1, [x2]
// CHECK-INST: ldbfmax h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c204041 <unknown>

ldbfmax h2, h3, [sp]
// CHECK-INST: ldbfmax h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c2243e3 <unknown>

// -- ldbfmaxa

ldbfmaxa h0, h1, [x2]
// CHECK-INST: ldbfmaxa h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xa0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca04041 <unknown>

ldbfmaxa h2, h3, [sp]
// CHECK-INST: ldbfmaxa h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xa2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca243e3 <unknown>

// -- ldbfmaxal

ldbfmaxal h0, h1, [x2]
// CHECK-INST: ldbfmaxal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0xe0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce04041 <unknown>

ldbfmaxal h2, h3, [sp]
// CHECK-INST: ldbfmaxal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0xe2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce243e3 <unknown>

// -- ldbfmaxl

ldbfmaxl h0, h1, [x2]
// CHECK-INST: ldbfmaxl h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x40,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c604041 <unknown>

ldbfmaxl h2, h3, [sp]
// CHECK-INST: ldbfmaxl h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x43,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c6243e3 <unknown>