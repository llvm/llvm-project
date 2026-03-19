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
// LDFADD
//------------------------------------------------------------------------------

ldfadd h0, h1, [x2]
// CHECK-INST: ldfadd h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c200041 <unknown>

ldfadd h2, h3, [sp]
// CHECK-INST: ldfadd h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c2203e3 <unknown>

ldfadd s0, s1, [x2]
// CHECK-INST: ldfadd s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc200041 <unknown>

ldfadd s2, s3, [sp]
// CHECK-INST: ldfadd s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc2203e3 <unknown>

ldfadd d0, d1, [x2]
// CHECK-INST: ldfadd d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc200041 <unknown>

ldfadd d2, d3, [sp]
// CHECK-INST: ldfadd d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc2203e3 <unknown>

// -- ldfadda

ldfadda h0, h1, [x2]
// CHECK-INST: ldfadda h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xa0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca00041 <unknown>

ldfadda h2, h3, [sp]
// CHECK-INST: ldfadda h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xa2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca203e3 <unknown>

ldfadda s0, s1, [x2]
// CHECK-INST: ldfadda s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xa0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca00041 <unknown>

ldfadda s2, s3, [sp]
// CHECK-INST: ldfadda s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xa2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca203e3 <unknown>

ldfadda d0, d1, [x2]
// CHECK-INST: ldfadda d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xa0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca00041 <unknown>

ldfadda d2, d3, [sp]
// CHECK-INST: ldfadda d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xa2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca203e3 <unknown>

// -- ldfaddal

ldfaddal h0, h1, [x2]
// CHECK-INST: ldfaddal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xe0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce00041 <unknown>

ldfaddal h2, h3, [sp]
// CHECK-INST: ldfaddal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xe2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce203e3 <unknown>

ldfaddal s0, s1, [x2]
// CHECK-INST: ldfaddal s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xe0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce00041 <unknown>

ldfaddal s2, s3, [sp]
// CHECK-INST: ldfaddal s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xe2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce203e3 <unknown>

ldfaddal d0, d1, [x2]
// CHECK-INST: ldfaddal d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xe0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce00041 <unknown>

ldfaddal d2, d3, [sp]
// CHECK-INST: ldfaddal d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xe2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce203e3 <unknown>

// -- ldfaddl

ldfaddl h0, h1, [x2]
// CHECK-INST: ldfaddl h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN:  7c600041 <unknown>

ldfaddl h2, h3, [sp]
// CHECK-INST: ldfaddl h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c6203e3 <unknown>

ldfaddl s0, s1, [x2]
// CHECK-INST: ldfaddl s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc600041 <unknown>

ldfaddl s2, s3, [sp]
// CHECK-INST: ldfaddl s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc6203e3 <unknown>

ldfaddl d0, d1, [x2]
// CHECK-INST: ldfaddl d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc600041 <unknown>

ldfaddl d2, d3, [sp]
// CHECK-INST: ldfaddl d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc6203e3 <unknown

//------------------------------------------------------------------------------
// LDBFADD
//------------------------------------------------------------------------------

ldbfadd h0, h1, [x2]
// CHECK-INST: ldbfadd h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c200041 <unknown>

ldbfadd h2, h3, [sp]
// CHECK-INST: ldbfadd h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c2203e3 <unknown>

// -- ldbfadda

ldbfadda h0, h1, [x2]
// CHECK-INST: ldbfadda h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xa0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca00041 <unknown>

ldbfadda h2, h3, [sp]
// CHECK-INST: ldbfadda h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xa2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca203e3 <unknown>

// -- ldbfaddal

ldbfaddal h0, h1, [x2]
// CHECK-INST: ldbfaddal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0xe0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce00041 <unknown>

ldbfaddal h2, h3, [sp]
// CHECK-INST: ldbfaddal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0xe2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce203e3 <unknown>

// -- ldbfaddl

ldbfaddl h0, h1, [x2]
// CHECK-INST: ldbfaddl h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x00,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c600041 <unknown>

ldbfaddl h2, h3, [sp]
// CHECK-INST: ldbfaddl h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x03,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c6203e3 <unknown>