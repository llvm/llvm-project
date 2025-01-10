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
// LDFMIN
//------------------------------------------------------------------------------

ldfmin h0, h1, [x2]
// CHECK-INST: ldfmin h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c205041 <unknown>

ldfmin h2, h3, [sp]
// CHECK-INST: ldfmin h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c2253e3 <unknown>

ldfmin s0, s1, [x2]
// CHECK-INST: ldfmin s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc205041 <unknown>

ldfmin s2, s3, [sp]
// CHECK-INST: ldfmin s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc2253e3 <unknown>

ldfmin d0, d1, [x2]
// CHECK-INST: ldfmin d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc205041 <unknown>

ldfmin d2, d3, [sp]
// CHECK-INST: ldfmin d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc2253e3 <unknown>

// -- ldfmina

ldfmina h0, h1, [x2]
// CHECK-INST: ldfmina h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xa0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca05041 <unknown>

ldfmina h2, h3, [sp]
// CHECK-INST: ldfmina h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xa2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ca253e3 <unknown>

ldfmina s0, s1, [x2]
// CHECK-INST: ldfmina s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xa0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca05041 <unknown>

ldfmina s2, s3, [sp]
// CHECK-INST: ldfmina s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xa2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bca253e3 <unknown>

ldfmina d0, d1, [x2]
// CHECK-INST: ldfmina d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xa0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca05041 <unknown>

ldfmina d2, d3, [sp]
// CHECK-INST: ldfmina d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xa2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fca253e3 <unknown>

// -- ldfminal

ldfminal h0, h1, [x2]
// CHECK-INST: ldfminal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xe0,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce05041 <unknown>

ldfminal h2, h3, [sp]
// CHECK-INST: ldfminal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xe2,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7ce253e3 <unknown>

ldfminal s0, s1, [x2]
// CHECK-INST: ldfminal s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xe0,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce05041 <unknown>

ldfminal s2, s3, [sp]
// CHECK-INST: ldfminal s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xe2,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bce253e3 <unknown>

ldfminal d0, d1, [x2]
// CHECK-INST: ldfminal d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xe0,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce05041 <unknown>

ldfminal d2, d3, [sp]
// CHECK-INST: ldfminal d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xe2,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fce253e3 <unknown>

// -- ldfminl

ldfminl h0, h1, [x2]
// CHECK-INST: ldfminl h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c605041 <unknown>

ldfminl h2, h3, [sp]
// CHECK-INST: ldfminl h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c6253e3 <unknown>

ldfminl s0, s1, [x2]
// CHECK-INST: ldfminl s0, s1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc605041 <unknown>

ldfminl s2, s3, [sp]
// CHECK-INST: ldfminl s2, s3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc6253e3 <unknown>

ldfminl d0, d1, [x2]
// CHECK-INST: ldfminl d0, d1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc605041 <unknown>

ldfminl d2, d3, [sp]
// CHECK-INST: ldfminl d2, d3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc6253e3 <unknown>

//------------------------------------------------------------------------------
// LDBFMIN
//------------------------------------------------------------------------------

ldbfmin h0, h1, [x2]
// CHECK-INST: ldbfmin h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c205041 <unknown>

ldbfmin h2, h3, [sp]
// CHECK-INST: ldbfmin h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c2253e3 <unknown>

// -- ldbfmina

ldbfmina h0, h1, [x2]
// CHECK-INST: ldbfmina h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xa0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca05041 <unknown>

ldbfmina h2, h3, [sp]
// CHECK-INST: ldbfmina h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xa2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ca253e3 <unknown>

// -- ldbfminal

ldbfminal h0, h1, [x2]
// CHECK-INST: ldbfminal h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0xe0,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce05041 <unknown>

ldbfminal h2, h3, [sp]
// CHECK-INST: ldbfminal h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0xe2,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3ce253e3 <unknown>

// -- ldbfminl

ldbfminl h0, h1, [x2]
// CHECK-INST: ldbfminl h0, h1, [x2]
// CHECK-ENCODING: [0x41,0x50,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c605041 <unknown>

ldbfminl h2, h3, [sp]
// CHECK-INST: ldbfminl h2, h3, [sp]
// CHECK-ENCODING: [0xe3,0x53,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c6253e3 <unknown>