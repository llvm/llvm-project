// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

// --------------------------------------------------------------------------//
// Test addressing modes

prfw    pldl1keep, p0, [x0, z0.s, uxtw #2]
// CHECK-INST: prfw    pldl1keep, p0, [x0, z0.s, uxtw #2]
// CHECK-ENCODING: [0x00,0x40,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84204000 <unknown>

prfw    pldl3strm, p5, [x10, z21.s, sxtw #2]
// CHECK-INST: prfw    pldl3strm, p5, [x10, z21.s, sxtw #2]
// CHECK-ENCODING: [0x45,0x55,0x75,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84755545 <unknown>

prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK-INST: prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK-ENCODING: [0xa7,0x4d,0x28,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4284da7 <unknown>

prfw    pldl1keep, p0, [x0, z0.d, sxtw #2]
// CHECK-INST: prfw    pldl1keep, p0, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0x40,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4604000 <unknown>

prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK-INST: prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK-ENCODING: [0x45,0xd5,0x75,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c475d545 <unknown>

prfw    #15, p7, [z31.s, #0]
// CHECK-INST: prfw    #15, p7, [z31.s]
// CHECK-ENCODING: [0xef,0xff,0x00,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 8500ffef <unknown>

prfw    #15, p7, [z31.s, #124]
// CHECK-INST: prfw    #15, p7, [z31.s, #124]
// CHECK-ENCODING: [0xef,0xff,0x1f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 851fffef <unknown>

prfw    #15, p7, [z31.d, #0]
// CHECK-INST: prfw    #15, p7, [z31.d]
// CHECK-ENCODING: [0xef,0xff,0x00,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c500ffef <unknown>

prfw    #15, p7, [z31.d, #124]
// CHECK-INST: prfw    #15, p7, [z31.d, #124]
// CHECK-ENCODING: [0xef,0xff,0x1f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c51fffef <unknown>
