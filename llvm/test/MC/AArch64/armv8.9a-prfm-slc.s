
// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s --check-prefix=NO-SLC
// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v8.9a < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v9.4a < %s | FileCheck %s

prfm pldslckeep, [x3]
// CHECK: prfm pldslckeep, [x3]  // encoding: [0x66,0x00,0x80,0xf9]
// NO-SLC: prfm #6, [x3]
prfm pldslcstrm, [x3]
// CHECK: prfm pldslcstrm, [x3]  // encoding: [0x67,0x00,0x80,0xf9]
// NO-SLC: prfm #7, [x3]
prfm plislckeep, [x3]
// CHECK: prfm plislckeep, [x3]  // encoding: [0x6e,0x00,0x80,0xf9]
// NO-SLC: prfm #14, [x3]
prfm plislcstrm, [x3]
// CHECK: prfm plislcstrm, [x3]  // encoding: [0x6f,0x00,0x80,0xf9]
// NO-SLC: prfm #15, [x3]
prfm pstslckeep, [x3]
// CHECK: prfm pstslckeep, [x3]  // encoding: [0x76,0x00,0x80,0xf9]
// NO-SLC: prfm #22, [x3]
prfm pstslcstrm, [x3]
// CHECK: prfm pstslcstrm, [x3]  // encoding: [0x77,0x00,0x80,0xf9]
// NO-SLC: prfm #23, [x3]

self:
prfm pldslckeep, self
// CHECK: prfm pldslckeep, self // encoding: [0bAAA00110,A,A,0xd8]
// NO-SLC: prfm #6, self

prfm pldslckeep, [x3, x5]
// CHECK: prfm pldslckeep, [x3, x5] // encoding: [0x66,0x68,0xa5,0xf8]
// NO-SLC: prfm #6, [x3, x5]
