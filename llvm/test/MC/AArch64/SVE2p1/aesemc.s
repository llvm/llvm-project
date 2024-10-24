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
aesemc  {z0.b-z1.b}, {z0.b-z1.b}, z0.q[0]  // 01000101-00100011-11101000-00000000
// CHECK-INST: aesemc  { z0.b, z1.b }, { z0.b, z1.b }, z0.q[0]
// CHECK-ENCODING: [0x00,0xe8,0x23,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4523e800 <unknown>

aesemc  {z22.b-z23.b}, {z22.b-z23.b}, z13.q[1]  // 01000101-00101011-11101001-10110110
// CHECK-INST: aesemc  { z22.b, z23.b }, { z22.b, z23.b }, z13.q[1]
// CHECK-ENCODING: [0xb6,0xe9,0x2b,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 452be9b6 <unknown>

aesemc  {z30.b-z31.b}, {z30.b-z31.b}, z31.q[3]  // 01000101-00111011-11101011-11111110
// CHECK-INST: aesemc  { z30.b, z31.b }, { z30.b, z31.b }, z31.q[3]
// CHECK-ENCODING: [0xfe,0xeb,0x3b,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453bebfe <unknown>

// x4
aesemc  {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]  // 01000101-00100111-11101000-00000000
// CHECK-INST: aesemc  { z0.b - z3.b }, { z0.b - z3.b }, z0.q[0]
// CHECK-ENCODING: [0x00,0xe8,0x27,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4527e800 <unknown>

aesemc  {z20.b-z23.b}, {z20.b-z23.b}, z10.q[2]  // 01000101-00110111-11101001-01010100
// CHECK-INST: aesemc  { z20.b - z23.b }, { z20.b - z23.b }, z10.q[2]
// CHECK-ENCODING: [0x54,0xe9,0x37,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4537e954 <unknown>

aesemc  {z28.b-z31.b}, {z28.b-z31.b}, z31.q[3]  // 01000101-00111111-11101011-11111100
// CHECK-INST: aesemc  { z28.b - z31.b }, { z28.b - z31.b }, z31.q[3]
// CHECK-ENCODING: [0xfc,0xeb,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453febfc <unknown>