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

prfd    pldl1keep, p0, [x0, z0.s, uxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.s, uxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 20 84 <unknown>

prfd    pldl1keep, p0, [x0, z0.s, sxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.s, sxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x60,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 60 84 <unknown>

prfd    pldl1keep, p0, [x0, z0.d, uxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.d, uxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x20,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 20 c4 <unknown>

prfd    pldl1keep, p0, [x0, z0.d, sxtw #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.d, sxtw #3]
// CHECK-ENCODING: [0x00,0x60,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 60 c4 <unknown>

prfd    pldl1keep, p0, [x0, z0.d, lsl #3]
// CHECK-INST: prfd    pldl1keep, p0, [x0, z0.d, lsl #3]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 60 c4 <unknown>

prfd    #15, p7, [z31.s, #0]
// CHECK-INST: prfd    #15, p7, [z31.s]
// CHECK-ENCODING: [0xef,0xff,0x80,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 80 85 <unknown>

prfd    #15, p7, [z31.s, #248]
// CHECK-INST: prfd    #15, p7, [z31.s, #248]
// CHECK-ENCODING: [0xef,0xff,0x9f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 9f 85 <unknown>

prfd    #15, p7, [z31.d, #0]
// CHECK-INST: prfd    #15, p7, [z31.d]
// CHECK-ENCODING: [0xef,0xff,0x80,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 80 c5 <unknown>

prfd    #15, p7, [z31.d, #248]
// CHECK-INST: prfd    #15, p7, [z31.d, #248]
// CHECK-ENCODING: [0xef,0xff,0x9f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 9f c5 <unknown>
