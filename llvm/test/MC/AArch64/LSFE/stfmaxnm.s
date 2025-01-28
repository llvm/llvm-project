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
// STFMAXNM
//------------------------------------------------------------------------------

stfmaxnm h0, [x2]
// CHECK-INST: stfmaxnm h0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x20,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c20e05f <unknown>

stfmaxnm h2, [sp]
// CHECK-INST: stfmaxnm h2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x22,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c22e3ff <unknown>

stfmaxnm s0, [x2]
// CHECK-INST: stfmaxnm s0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x20,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc20e05f <unknown>

stfmaxnm s2, [sp]
// CHECK-INST: stfmaxnm s2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x22,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc22e3ff <unknown>

stfmaxnm d0, [x2]
// CHECK-INST: stfmaxnm d0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x20,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc20e05f <unknown>

stfmaxnm d2, [sp]
// CHECK-INST: stfmaxnm d2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x22,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc22e3ff <unknown>

// -- stfmaxnml

stfmaxnml h0, [x2]
// CHECK-INST: stfmaxnml h0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x60,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN:  7c60e05f <unknown>

stfmaxnml h2, [sp]
// CHECK-INST: stfmaxnml h2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x62,0x7c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 7c62e3ff <unknown>

stfmaxnml s0, [x2]
// CHECK-INST: stfmaxnml s0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x60,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc60e05f <unknown>

stfmaxnml s2, [sp]
// CHECK-INST: stfmaxnml s2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x62,0xbc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: bc62e3ff <unknown>

stfmaxnml d0, [x2]
// CHECK-INST: stfmaxnml d0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x60,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc60e05f <unknown>

stfmaxnml d2, [sp]
// CHECK-INST: stfmaxnml d2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x62,0xfc]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: fc62e3ff <unknown>

//------------------------------------------------------------------------------
// STBFMAXNM
//------------------------------------------------------------------------------

stbfmaxnm h0, [x2]
// CHECK-INST: stbfmaxnm h0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x20,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c20e05f <unknown>

stbfmaxnm h2, [sp]
// CHECK-INST: stbfmaxnm h2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x22,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c22e3ff <unknown>

// -- stbfmaxnml

stbfmaxnml h0, [x2]
// CHECK-INST: stbfmaxnml h0, [x2]
// CHECK-ENCODING: [0x5f,0xe0,0x60,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c60e05f <unknown>

stbfmaxnml h2, [sp]
// CHECK-INST: stbfmaxnml h2, [sp]
// CHECK-ENCODING: [0xff,0xe3,0x62,0x3c]
// CHECK-ERROR: instruction requires: lsfe
// CHECK-UNKNOWN: 3c62e3ff <unknown>
