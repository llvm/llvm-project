// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f8f32,+sme-f8f16,+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f8f32,+sme-f8f16,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-f8f32,+sme-f8f16,+sme-f16f16 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f8f32,+sme-f8f16,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f8f32,+sme-f8f16,+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-f8f32,+sme-f8f16,+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// 2-way (fp8-to-fp16)

ftmopa  za0.h, {z0.b-z1.b}, z0.b, z20[0]  // 10000000-01100000-00000000-00001000
// CHECK-INST: ftmopa  za0.h, { z0.b, z1.b }, z0.b, z20[0]
// CHECK-ENCODING: [0x08,0x00,0x60,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f16
// CHECK-UNKNOWN: 80600008 <unknown>

ftmopa  za1.h, {z10.b-z11.b}, z21.b, z29[1]  // 10000000-01110101-00010101-01011001
// CHECK-INST: ftmopa  za1.h, { z10.b, z11.b }, z21.b, z29[1]
// CHECK-ENCODING: [0x59,0x15,0x75,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f16
// CHECK-UNKNOWN: 80751559 <unknown>

ftmopa  za1.h, {z30.b-z31.b}, z31.b, z31[3]  // 10000000-01111111-00011111-11111001
// CHECK-INST: ftmopa  za1.h, { z30.b, z31.b }, z31.b, z31[3]
// CHECK-ENCODING: [0xf9,0x1f,0x7f,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f16
// CHECK-UNKNOWN: 807f1ff9 <unknown>

// 2-way, (fp16-to-fp32)

ftmopa  za0.s, {z0.h-z1.h}, z0.h, z20[0]  // 10000001-01100000-00000000-00000000
// CHECK-INST: ftmopa  za0.s, { z0.h, z1.h }, z0.h, z20[0]
// CHECK-ENCODING: [0x00,0x00,0x60,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81600000 <unknown>

ftmopa  za3.s, {z12.h-z13.h}, z8.h, z23[3]  // 10000001-01101000-00001101-10110011
// CHECK-INST: ftmopa  za3.s, { z12.h, z13.h }, z8.h, z23[3]
// CHECK-ENCODING: [0xb3,0x0d,0x68,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81680db3 <unknown>

ftmopa  za3.s, {z30.h-z31.h}, z31.h, z31[3]  // 10000001-01111111-00011111-11110011
// CHECK-INST: ftmopa  za3.s, { z30.h, z31.h }, z31.h, z31[3]
// CHECK-ENCODING: [0xf3,0x1f,0x7f,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 817f1ff3 <unknown>

// 4-way

ftmopa  za0.s, {z0.b-z1.b}, z0.b, z20[0]  // 10000000-01100000-00000000-00000000
// CHECK-INST: ftmopa  za0.s, { z0.b, z1.b }, z0.b, z20[0]
// CHECK-ENCODING: [0x00,0x00,0x60,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80600000 <unknown>

ftmopa  za3.s, {z12.b-z13.b}, z8.b, z23[3]  // 10000000-01101000-00001101-10110011
// CHECK-INST: ftmopa  za3.s, { z12.b, z13.b }, z8.b, z23[3]
// CHECK-ENCODING: [0xb3,0x0d,0x68,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 80680db3 <unknown>

ftmopa  za3.s, {z30.b-z31.b}, z31.b, z31[3]  // 10000000-01111111-00011111-11110011
// CHECK-INST: ftmopa  za3.s, { z30.b, z31.b }, z31.b, z31[3]
// CHECK-ENCODING: [0xf3,0x1f,0x7f,0x80]
// CHECK-ERROR: instruction requires: sme2p2 sme-f8f32
// CHECK-UNKNOWN: 807f1ff3 <unknown>

// non-widening (half-precision)

ftmopa  za0.h, {z0.h-z1.h}, z0.h, z20[0]  // 10000001-01000000-00000000-00001000
// CHECK-INST: ftmopa  za0.h, { z0.h, z1.h }, z0.h, z20[0]
// CHECK-ENCODING: [0x08,0x00,0x40,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81400008 <unknown>

ftmopa  za1.h, {z12.h-z13.h}, z8.h, z23[3]  // 10000001-01001000-00001101-10111001
// CHECK-INST: ftmopa  za1.h, { z12.h, z13.h }, z8.h, z23[3]
// CHECK-ENCODING: [0xb9,0x0d,0x48,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81480db9 <unknown>

ftmopa  za1.h, {z30.h-z31.h}, z31.h, z31[3]  // 10000001-01011111-00011111-11111011
// CHECK-INST: ftmopa  za1.h, { z30.h, z31.h }, z31.h, z31[3]
// CHECK-ENCODING: [0xf9,0x1f,0x5f,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 815f1ff9 <unknown>

// non-widening (single-precision)

ftmopa  za0.s, {z0.s-z1.s}, z0.s, z20[0]  // 10000000-01000000-00000000-00000000
// CHECK-INST: ftmopa  za0.s, { z0.s, z1.s }, z0.s, z20[0]
// CHECK-ENCODING: [0x00,0x00,0x40,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80400000 <unknown>

ftmopa  za3.s, {z12.s-z13.s}, z8.s, z23[3]  // 10000000-01001000-00001101-10110011
// CHECK-INST: ftmopa  za3.s, { z12.s, z13.s }, z8.s, z23[3]
// CHECK-ENCODING: [0xb3,0x0d,0x48,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 80480db3 <unknown>

ftmopa  za3.s, {z30.s-z31.s}, z31.s, z31[3]  // 10000000-01011111-00011111-11110011
// CHECK-INST: ftmopa  za3.s, { z30.s, z31.s }, z31.s, z31[3]
// CHECK-ENCODING: [0xf3,0x1f,0x5f,0x80]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 805f1ff3 <unknown>