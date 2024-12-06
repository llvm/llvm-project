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
// LDFMAXNM
//------------------------------------------------------------------------------

ldfmaxnm h0, h1, [x2]
// CHECK-INST: ldfmaxnm h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c206041 <unknown>

ldfmaxnm h2, h3, [sp]
// CHECK-INST: ldfmaxnm h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c2263e3 <unknown>

ldfmaxnm s0, s1, [x2]
// CHECK-INST: ldfmaxnm s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc206041 <unknown>

ldfmaxnm s2, s3, [sp]
// CHECK-INST: ldfmaxnm s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc2263e3 <unknown>

ldfmaxnm d0, d1, [x2]
// CHECK-INST: ldfmaxnm d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc206041 <unknown>

ldfmaxnm d2, d3, [sp]
// CHECK-INST: ldfmaxnm d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc2263e3 <unknown>

// -- ldfmaxnma

ldfmaxnma h0, h1, [x2]
// CHECK-INST: ldfmaxnma h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xa0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca06041 <unknown>

ldfmaxnma h2, h3, [sp]
// CHECK-INST: ldfmaxnma h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xa2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca263e3 <unknown>

ldfmaxnma s0, s1, [x2]
// CHECK-INST: ldfmaxnma s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xa0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca06041 <unknown>

ldfmaxnma s2, s3, [sp]
// CHECK-INST: ldfmaxnma s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xa2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca263e3 <unknown>

ldfmaxnma d0, d1, [x2]
// CHECK-INST: ldfmaxnma d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xa0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca06041 <unknown>

ldfmaxnma d2, d3, [sp]
// CHECK-INST: ldfmaxnma d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xa2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca263e3 <unknown>

// -- ldfmaxnmal

ldfmaxnmal h0, h1, [x2]
// CHECK-INST: ldfmaxnmal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xe0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce06041 <unknown>

ldfmaxnmal h2, h3, [sp]
// CHECK-INST: ldfmaxnmal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xe2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce263e3 <unknown>

ldfmaxnmal s0, s1, [x2]
// CHECK-INST: ldfmaxnmal s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xe0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce06041 <unknown>

ldfmaxnmal s2, s3, [sp]
// CHECK-INST: ldfmaxnmal s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xe2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce263e3 <unknown>

ldfmaxnmal d0, d1, [x2]
// CHECK-INST: ldfmaxnmal d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xe0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce06041 <unknown>

ldfmaxnmal d2, d3, [sp]
// CHECK-INST: ldfmaxnmal d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xe2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce263e3 <unknown>

// -- ldfmaxnml

ldfmaxnml h0, h1, [x2]
// CHECK-INST: ldfmaxnml h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN:  7c606041 <unknown>

ldfmaxnml h2, h3, [sp]
// CHECK-INST: ldfmaxnml h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c6263e3 <unknown>

ldfmaxnml s0, s1, [x2]
// CHECK-INST: ldfmaxnml s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc606041 <unknown>

ldfmaxnml s2, s3, [sp]
// CHECK-INST: ldfmaxnml s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc6263e3 <unknown>

ldfmaxnml d0, d1, [x2]
// CHECK-INST: ldfmaxnml d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc606041 <unknown>

ldfmaxnml d2, d3, [sp]
// CHECK-INST: ldfmaxnml d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc6263e3 <unknown>

//------------------------------------------------------------------------------
// LDBFMAXNM
//------------------------------------------------------------------------------

ldbfmaxnm h0, h1, [x2]
// CHECK-INST: ldbfmaxnm h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c206041 <unknown>

ldbfmaxnm h2, h3, [sp]
// CHECK-INST: ldbfmaxnm h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c2263e3 <unknown>

// -- ldbfmaxnma

ldbfmaxnma h0, h1, [x2]
// CHECK-INST: ldbfmaxnma h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xa0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca06041 <unknown>

ldbfmaxnma h2, h3, [sp]
// CHECK-INST: ldbfmaxnma h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xa2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca263e3 <unknown>

// -- ldbfmaxnmal

ldbfmaxnmal h0, h1, [x2]
// CHECK-INST: ldbfmaxnmal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0xe0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce06041 <unknown>

ldbfmaxnmal h2, h3, [sp]
// CHECK-INST: ldbfmaxnmal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0xe2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce263e3 <unknown>

// -- ldbfmaxnml

ldbfmaxnml h0, h1, [x2]
// CHECK-INST: ldbfmaxnml h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x60,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c606041 <unknown>

ldbfmaxnml h2, h3, [sp]
// CHECK-INST: ldbfmaxnml h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x63,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c6263e3 <unknown>