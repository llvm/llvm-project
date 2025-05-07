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
// STFMAX
//------------------------------------------------------------------------------

stfmax h0, [x2]
// CHECK-INST: stfmax h0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c20c05f <unknown>

stfmax h2, [sp]
// CHECK-INST: stfmax h2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c22c3ff <unknown>

stfmax s0, [x2]
// CHECK-INST: stfmax s0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc20c05f <unknown>

stfmax s2, [sp]
// CHECK-INST: stfmax s2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc22c3ff <unknown>

stfmax d0, [x2]
// CHECK-INST: stfmax d0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc20c05f <unknown>

stfmax d2, [sp]
// CHECK-INST: stfmax d2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc22c3ff <unknown>

// -- stfmaxl

stfmaxl h0, [x2]
// CHECK-INST: stfmaxl h0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c60c05f <unknown>

stfmaxl h2, [sp]
// CHECK-INST: stfmaxl h2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c62c3ff <unknown>

stfmaxl s0, [x2]
// CHECK-INST: stfmaxl s0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc60c05f <unknown>

stfmaxl s2, [sp]
// CHECK-INST: stfmaxl s2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc62c3ff <unknown>

stfmaxl d0, [x2]
// CHECK-INST: stfmaxl d0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc60c05f <unknown>

stfmaxl d2, [sp]
// CHECK-INST: stfmaxl d2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc62c3ff <unknown>

//------------------------------------------------------------------------------
// STBFMAX
//------------------------------------------------------------------------------

stbfmax h0, [x2]
// CHECK-INST: stbfmax h0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c20c05f <unknown>

stbfmax h2, [sp]
// CHECK-INST: stbfmax h2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c22c3ff <unknown>

// -- stbfmaxl

stbfmaxl h0, [x2]
// CHECK-INST: stbfmaxl h0, [x2]
// CHECK-ENCODING: [0x5f,0xc0,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c60c05f <unknown>

stbfmaxl h2, [sp]
// CHECK-INST: stbfmaxl h2, [sp]
// CHECK-ENCODING: [0xff,0xc3,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c62c3ff <unknown>