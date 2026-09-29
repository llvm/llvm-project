// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

st4q    {z0.q, z1.q, z2.q, z3.q}, p0, [x0, x0, lsl #4]  // 11100100-11100000-00000000-00000000
// CHECK-INST: st4q    { z0.q - z3.q }, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4e00000 <unknown>

st4q    {z21.q, z22.q, z23.q, z24.q}, p5, [x10, x21, lsl #4]  // 11100100-11110101-00010101-01010101
// CHECK-INST: st4q    { z21.q - z24.q }, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x55,0x15,0xf5,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4f51555 <unknown>

st4q    {z23.q, z24.q, z25.q, z26.q}, p3, [x13, x8, lsl #4]  // 11100100-11101000-00001101-10110111
// CHECK-INST: st4q    { z23.q - z26.q }, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xb7,0x0d,0xe8,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4e80db7 <unknown>

st4q    {z0.q, z1.q, z2.q, z3.q}, p0, [x0]  // 11100100-11000000-00000000-00000000
// CHECK-INST: st4q    { z0.q - z3.q }, p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4c00000 <unknown>

st4q    {z21.q, z22.q, z23.q, z24.q}, p5, [x10, #20, mul vl]  // 11100100-11000101-00010101-01010101
// CHECK-INST: st4q    { z21.q - z24.q }, p5, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0x15,0xc5,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4c51555 <unknown>

st4q    {z23.q, z24.q, z25.q, z26.q}, p3, [x13, #-32, mul vl]  // 11100100-11001000-00001101-10110111
// CHECK-INST: st4q    { z23.q - z26.q }, p3, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0x0d,0xc8,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4c80db7 <unknown>

st4q    {z31.q, z0.q, z1.q, z2.q}, p7, [sp, #-4, mul vl]  // 11100100-11001111-00011111-11111111
// CHECK-INST: st4q    { z31.q, z0.q, z1.q, z2.q }, p7, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xff,0x1f,0xcf,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4cf1fff <unknown>

