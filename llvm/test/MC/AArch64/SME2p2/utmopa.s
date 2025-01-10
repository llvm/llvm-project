// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

utmopa  za0.s, {z0.h-z1.h}, z0.h, z20[0]  // 10000001-01000000-10000000-00001000
// CHECK-INST: utmopa  za0.s, { z0.h, z1.h }, z0.h, z20[0]
// CHECK-ENCODING: [0x08,0x80,0x40,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81408008 <unknown>

utmopa  za3.s, {z12.h-z13.h}, z8.h, z23[3]  // 10000001-01001000-10001101-10111011
// CHECK-INST: utmopa  za3.s, { z12.h, z13.h }, z8.h, z23[3]
// CHECK-ENCODING: [0xbb,0x8d,0x48,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81488dbb <unknown>

utmopa  za3.s, {z30.h-z31.h}, z31.h, z31[3]  // 10000001-01011111-10011111-11111011
// CHECK-INST: utmopa  za3.s, { z30.h, z31.h }, z31.h, z31[3]
// CHECK-ENCODING: [0xfb,0x9f,0x5f,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 815f9ffb <unknown>

utmopa  za0.s, {z0.b-z1.b}, z0.b, z20[0]  // 10000001-01100000-10000000-00000000
// CHECK-INST: utmopa  za0.s, { z0.b, z1.b }, z0.b, z20[0]
// CHECK-ENCODING: [0x00,0x80,0x60,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81608000 <unknown>

utmopa  za3.s, {z12.b-z13.b}, z8.b, z23[3]  // 10000001-01101000-10001101-10110011
// CHECK-INST: utmopa  za3.s, { z12.b, z13.b }, z8.b, z23[3]
// CHECK-ENCODING: [0xb3,0x8d,0x68,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81688db3 <unknown>

utmopa  za3.s, {z30.b-z31.b}, z31.b, z31[3]  // 10000001-01111111-10011111-11110011
// CHECK-INST: utmopa  za3.s, { z30.b, z31.b }, z31.b, z31[3]
// CHECK-ENCODING: [0xf3,0x9f,0x7f,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 817f9ff3 <unknown>
