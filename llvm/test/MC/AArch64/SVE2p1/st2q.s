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

st2q    {z0.q, z1.q}, p0, [x0, x0, lsl #4]  // 11100100-01100000-00000000-00000000
// CHECK-INST: st2q    { z0.q, z1.q }, p0, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x00,0x60,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4600000 <unknown>

st2q    {z21.q, z22.q}, p5, [x10, x21, lsl #4]  // 11100100-01110101-00010101-01010101
// CHECK-INST: st2q    { z21.q, z22.q }, p5, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x55,0x15,0x75,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4751555 <unknown>

st2q    {z23.q, z24.q}, p3, [x13, x8, lsl #4]  // 11100100-01101000-00001101-10110111
// CHECK-INST: st2q    { z23.q, z24.q }, p3, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xb7,0x0d,0x68,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4680db7 <unknown>

st2q    {z0.q, z1.q}, p0, [x0]  // 11100100-01000000-00000000-00000000
// CHECK-INST: st2q    { z0.q, z1.q }, p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0x40,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4400000 <unknown>

st2q    {z21.q, z22.q}, p5, [x10, #10, mul vl]  // 11100100-01000101-00010101-01010101
// CHECK-INST: st2q    { z21.q, z22.q }, p5, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x15,0x45,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4451555 <unknown>

st2q    {z23.q, z24.q}, p3, [x13, #-16, mul vl]  // 11100100-01001000-00001101-10110111
// CHECK-INST: st2q    { z23.q, z24.q }, p3, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x0d,0x48,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e4480db7 <unknown>

st2q    {z31.q, z0.q}, p7, [sp, #-2, mul vl]  // 11100100-01001111-00011111-11111111
// CHECK-INST: st2q    { z31.q, z0.q }, p7, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x1f,0x4f,0xe4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: e44f1fff <unknown>

