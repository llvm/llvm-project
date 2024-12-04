
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
aesdimc {z0.b-z1.b}, {z0.b-z1.b}, z0.q[0]  // 01000101-00100011-11101100-00000000
// CHECK-INST: aesdimc { z0.b, z1.b }, { z0.b, z1.b }, z0.q[0]
// CHECK-ENCODING: [0x00,0xec,0x23,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4523ec00 <unknown>

aesdimc {z20.b-z21.b}, {z20.b-z21.b}, z10.q[2]  // 01000101-00110011-11101101-01010100
// CHECK-INST: aesdimc { z20.b, z21.b }, { z20.b, z21.b }, z10.q[2]
// CHECK-ENCODING: [0x54,0xed,0x33,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4533ed54 <unknown>

aesdimc {z30.b-z31.b}, {z30.b-z31.b}, z31.q[3]  // 01000101-00111011-11101111-11111110
// CHECK-INST: aesdimc { z30.b, z31.b }, { z30.b, z31.b }, z31.q[3]
// CHECK-ENCODING: [0xfe,0xef,0x3b,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453beffe <unknown>

// x4
aesdimc {z0.b-z3.b}, {z0.b-z3.b}, z0.q[0]  // 01000101-00100111-11101100-00000000
// CHECK-INST: aesdimc { z0.b - z3.b }, { z0.b - z3.b }, z0.q[0]
// CHECK-ENCODING: [0x00,0xec,0x27,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 4527ec00 <unknown>

aesdimc {z20.b-z23.b}, {z20.b-z23.b}, z13.q[1]  // 01000101-00101111-11101101-10110100
// CHECK-INST: aesdimc { z20.b - z23.b }, { z20.b - z23.b }, z13.q[1]
// CHECK-ENCODING: [0xb4,0xed,0x2f,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 452fedb4 <unknown>

aesdimc {z28.b-z31.b}, {z28.b-z31.b}, z31.q[3]  // 01000101-00111111-11101111-11111100
// CHECK-INST: aesdimc { z28.b - z31.b }, { z28.b - z31.b }, z31.q[3]
// CHECK-ENCODING: [0xfc,0xef,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p1 or ssve-aes sve-aes2
// CHECK-UNKNOWN: 453feffc <unknown>