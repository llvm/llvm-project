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

st3q    {z0.q, z1.q, z2.q}, p0, [x0, x0, lsl #4]  // 11100100-10100000-00000000-00000000
// CHECK-INST: st3q    { z0.q - z2.q }, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4a00000 <unknown>

st3q    {z21.q, z22.q, z23.q}, p5, [x10, x21, lsl #4]  // 11100100-10110101-00010101-01010101
// CHECK-INST: st3q    { z21.q - z23.q }, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x55,0x15,0xb5,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4b51555 <unknown>

st3q    {z23.q, z24.q, z25.q}, p3, [x13, x8, lsl #4]  // 11100100-10101000-00001101-10110111
// CHECK-INST: st3q    { z23.q - z25.q }, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xb7,0x0d,0xa8,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4a80db7 <unknown>

st3q    {z0.q, z1.q, z2.q}, p0, [x0]  // 11100100-10000000-00000000-00000000
// CHECK-INST: st3q    { z0.q - z2.q }, p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0x80,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4800000 <unknown>

st3q    {z21.q, z22.q, z23.q}, p5, [x10, #15, mul vl]  // 11100100-10000101-00010101-01010101
// CHECK-INST: st3q    { z21.q - z23.q }, p5, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0x15,0x85,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4851555 <unknown>

st3q    {z23.q, z24.q, z25.q}, p3, [x13, #-24, mul vl]  // 11100100-10001000-00001101-10110111
// CHECK-INST: st3q    { z23.q - z25.q }, p3, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0x0d,0x88,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4880db7 <unknown>

st3q    {z31.q, z0.q, z1.q}, p7, [sp, #-3, mul vl]  // 11100100-10001111-00011111-11111111
// CHECK-INST: st3q    { z31.q, z0.q, z1.q }, p7, [sp, #-3, mul vl]
// CHECK-ENCODING: [0xff,0x1f,0x8f,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e48f1fff <unknown>
