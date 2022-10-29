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

prfh    pldl3strm, p5, [x10, z21.s, uxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.s, uxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x35,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84353545 <unknown>

prfh    pldl3strm, p5, [x10, z21.s, sxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.s, sxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x75,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84753545 <unknown>

prfh    pldl3strm, p5, [x10, z21.d, uxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.d, uxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x35,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4353545 <unknown>

prfh    pldl3strm, p5, [x10, z21.d, sxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.d, sxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x75,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4753545 <unknown>

prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK-INST: prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK-ENCODING: [0x00,0xa0,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c460a000 <unknown>

prfh    #15, p7, [z31.s, #0]
// CHECK-INST: prfh    #15, p7, [z31.s]
// CHECK-ENCODING: [0xef,0xff,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 8480ffef <unknown>

prfh    #15, p7, [z31.s, #62]
// CHECK-INST: prfh    #15, p7, [z31.s, #62]
// CHECK-ENCODING: [0xef,0xff,0x9f,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 849fffef <unknown>

prfh    #15, p7, [z31.d, #0]
// CHECK-INST: prfh    #15, p7, [z31.d]
// CHECK-ENCODING: [0xef,0xff,0x80,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c480ffef <unknown>

prfh    #15, p7, [z31.d, #62]
// CHECK-INST: prfh    #15, p7, [z31.d, #62]
// CHECK-ENCODING: [0xef,0xff,0x9f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c49fffef <unknown>
