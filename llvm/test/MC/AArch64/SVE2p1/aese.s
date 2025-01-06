// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+ssve-aes < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve-aes2,+sve2p1 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-aes2,+sve2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve-aes2,+sve2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// x2
aese    {z0.b-z1.b}, {z0.b-z1.b}, z0.q[0]  // 01000101-00100010-11101000-00000000
// CHECK-INST: aese    { z0.b, z1.b }, { z0.b, z1.b }, z0.q[0]
// CHECK-ENCODING: [0x00,0xe8,0x22,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4522e800 <unknown>

aese    {z20.b-z21.b}, {z20.b-z21.b}, z10.q[2]  // 01000101-00110010-11101001-01010100
// CHECK-INST: aese    { z20.b, z21.b }, { z20.b, z21.b }, z10.q[2]
// CHECK-ENCODING: [0x54,0xe9,0x32,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4532e954 <unknown>

aese    {z30.b-z31.b}, {z30.b-z31.b}, z31.q[3]  // 01000101-00111010-11101011-11111110
// CHECK-INST: aese    { z30.b, z31.b }, { z30.b, z31.b }, z31.q[3]
// CHECK-ENCODING: [0xfe,0xeb,0x3a,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453aebfe <unknown>

// x4
aese    {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]  // 01000101-00100110-11101000-00000000
// CHECK-INST: aese    { z0.b - z3.b }, { z0.b - z3.b }, z0.q[0]
// CHECK-ENCODING: [0x00,0xe8,0x26,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4526e800 <unknown>

aese    {z20.b-z23.b}, {z20.b-z23.b}, z13.q[1]  // 01000101-00101110-11101001-10110100
// CHECK-INST: aese    { z20.b - z23.b }, { z20.b - z23.b }, z13.q[1]
// CHECK-ENCODING: [0xb4,0xe9,0x2e,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 452ee9b4 <unknown>

aese    {z28.b-z31.b}, {z28.b-z31.b}, z31.q[3]  // 01000101-00111110-11101011-11111100
// CHECK-INST: aese    { z28.b - z31.b }, { z28.b - z31.b }, z31.q[3]
// CHECK-ENCODING: [0xfc,0xeb,0x3e,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453eebfc <unknown>