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

prfd    pldl1keep, p0, [x0, z0.s, uxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.s, uxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84206000 <unknown>

prfd    pldl1keep, p0, [x0, z0.s, sxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.s, sxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x60,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84606000 <unknown>

prfd    pldl1keep, p0, [x0, z0.d, uxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.d, uxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x20,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4206000 <unknown>

prfd    pldl1keep, p0, [x0, z0.d, sxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.d, sxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4606000 <unknown>

prfd    pldl1keep, p0, [x0, z0.d, lsl #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.d, lsl #3]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c460e000 <unknown>

prfd    #15, p7, [z31.s, #0]
// CHECK-INST: prfd    #15, p7, [z31.s]
// CHECK-ENCODING: [0xef,0xff,0x80,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 8580ffef <unknown>

prfd    #15, p7, [z31.s, #248]
// CHECK-INST: prfd    #15, p7, [z31.s, #248]
// CHECK-ENCODING: [0xef,0xff,0x9f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 859fffef <unknown>

prfd    #15, p7, [z31.d, #0]
// CHECK-INST: prfd    #15, p7, [z31.d]
// CHECK-ENCODING: [0xef,0xff,0x80,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c580ffef <unknown>

prfd    #15, p7, [z31.d, #248]
// CHECK-INST: prfd    #15, p7, [z31.d, #248]
// CHECK-ENCODING: [0xef,0xff,0x9f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c59fffef <unknown>
