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
// LDFMINNM
//------------------------------------------------------------------------------

ldfminnm h0, h1, [x2]
// CHECK-INST: ldfminnm h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c207041 <unknown>

ldfminnm h2, h3, [sp]
// CHECK-INST: ldfminnm h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c2273e3 <unknown>

ldfminnm s0, s1, [x2]
// CHECK-INST: ldfminnm s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc207041 <unknown>

ldfminnm s2, s3, [sp]
// CHECK-INST: ldfminnm s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc2273e3 <unknown>

ldfminnm d0, d1, [x2]
// CHECK-INST: ldfminnm d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc207041 <unknown>

ldfminnm d2, d3, [sp]
// CHECK-INST: ldfminnm d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc2273e3 <unknown>

// -- ldfminnma

ldfminnma h0, h1, [x2]
// CHECK-INST: ldfminnma h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xa0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca07041 <unknown>

ldfminnma h2, h3, [sp]
// CHECK-INST: ldfminnma h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xa2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca273e3 <unknown>

ldfminnma s0, s1, [x2]
// CHECK-INST: ldfminnma s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xa0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca07041 <unknown>

ldfminnma s2, s3, [sp]
// CHECK-INST: ldfminnma s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xa2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca273e3 <unknown>

ldfminnma d0, d1, [x2]
// CHECK-INST: ldfminnma d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xa0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca07041 <unknown>

ldfminnma d2, d3, [sp]
// CHECK-INST: ldfminnma d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xa2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca273e3 <unknown>

// -- ldfminnmal

ldfminnmal h0, h1, [x2]
// CHECK-INST: ldfminnmal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xe0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce07041 <unknown>

ldfminnmal h2, h3, [sp]
// CHECK-INST: ldfminnmal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xe2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce273e3 <unknown>

ldfminnmal s0, s1, [x2]
// CHECK-INST: ldfminnmal s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xe0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce07041 <unknown>

ldfminnmal s2, s3, [sp]
// CHECK-INST: ldfminnmal s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xe2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce273e3 <unknown>

ldfminnmal d0, d1, [x2]
// CHECK-INST: ldfminnmal d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xe0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce07041 <unknown>

ldfminnmal d2, d3, [sp]
// CHECK-INST: ldfminnmal d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xe2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce273e3 <unknown>

// -- ldfminnml

ldfminnml h0, h1, [x2]
// CHECK-INST: ldfminnml h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN:  7c607041 <unknown>

ldfminnml h2, h3, [sp]
// CHECK-INST: ldfminnml h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c6273e3 <unknown>

ldfminnml s0, s1, [x2]
// CHECK-INST: ldfminnml s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc607041 <unknown>

ldfminnml s2, s3, [sp]
// CHECK-INST: ldfminnml s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc6273e3 <unknown>

ldfminnml d0, d1, [x2]
// CHECK-INST: ldfminnml d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc607041 <unknown>

ldfminnml d2, d3, [sp]
// CHECK-INST: ldfminnml d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc6273e3 <unknown>

//------------------------------------------------------------------------------
// LDBFMINNM
//------------------------------------------------------------------------------

ldbfminnm h0, h1, [x2]
// CHECK-INST: ldbfminnm h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c207041 <unknown>

ldbfminnm h2, h3, [sp]
// CHECK-INST: ldbfminnm h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c2273e3 <unknown>

// -- ldbfminnma

ldbfminnma h0, h1, [x2]
// CHECK-INST: ldbfminnma h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xa0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca07041 <unknown>

ldbfminnma h2, h3, [sp]
// CHECK-INST: ldbfminnma h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xa2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca273e3 <unknown>

// -- ldbfminnmal

ldbfminnmal h0, h1, [x2]
// CHECK-INST: ldbfminnmal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0xe0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce07041 <unknown>

ldbfminnmal h2, h3, [sp]
// CHECK-INST: ldbfminnmal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0xe2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce273e3 <unknown>

// -- ldbfminnml

ldbfminnml h0, h1, [x2]
// CHECK-INST: ldbfminnml h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x70,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c607041 <unknown>

ldbfminnml h2, h3, [sp]
// CHECK-INST: ldbfminnml h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x73,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c6273e3 <unknown>