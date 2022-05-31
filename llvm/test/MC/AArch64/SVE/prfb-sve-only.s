// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

// --------------------------------------------------------------------------//
// Test addressing modes

prfb    pldl1keep, p0, [x0, z0.s, uxtw]
// CHECK-INST: prfb    pldl1keep, p0, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x00,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 20 84 <unknown>

prfb    pldl3strm, p5, [x10, z21.s, uxtw]
// CHECK-INST: prfb    pldl3strm, p5, [x10, z21.s, uxtw]
// CHECK-ENCODING: [0x45,0x15,0x35,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 15 35 84 <unknown>

prfb    pldl1keep, p0, [x0, z0.d, uxtw]
// CHECK-INST: prfb    pldl1keep, p0, [x0, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0x00,0x20,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 20 c4 <unknown>

prfb    pldl3strm, p5, [x10, z21.d, sxtw]
// CHECK-INST: prfb    pldl3strm, p5, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x45,0x15,0x75,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 15 75 c4 <unknown>

prfb    pldl1keep, p0, [x0, z0.d]
// CHECK-INST: prfb    pldl1keep, p0, [x0, z0.d]
// CHECK-ENCODING: [0x00,0x80,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 60 c4 <unknown>

prfb    #7, p3, [z13.s, #0]
// CHECK-INST: prfb    #7, p3, [z13.s]
// CHECK-ENCODING: [0xa7,0xed,0x00,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 ed 00 84 <unknown>

prfb    #7, p3, [z13.s, #31]
// CHECK-INST: prfb    #7, p3, [z13.s, #31]
// CHECK-ENCODING: [0xa7,0xed,0x1f,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 ed 1f 84 <unknown>

prfb    pldl3strm, p5, [z10.d, #0]
// CHECK-INST: prfb    pldl3strm, p5, [z10.d]
// CHECK-ENCODING: [0x45,0xf5,0x00,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 f5 00 c4 <unknown>

prfb    pldl3strm, p5, [z10.d, #31]
// CHECK-INST: prfb    pldl3strm, p5, [z10.d, #31]
// CHECK-ENCODING: [0x45,0xf5,0x1f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 f5 1f c4 <unknown>
