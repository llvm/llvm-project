// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-tmop < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-tmop < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-tmop --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-tmop < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-tmop --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-tmop < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-tmop -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

stmopa  za0.s, {z0.b-z1.b}, z0.b, z20[0]  // 10000000-01000000-10000000-00000000
// CHECK-INST: stmopa  za0.s, { z0.b, z1.b }, z0.b, z20[0]
// CHECK-ENCODING: [0x00,0x80,0x40,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 80408000 <unknown>

stmopa  za3.s, {z12.b-z13.b}, z8.b, z23[3]  // 10000000-01001000-10001101-10110011
// CHECK-INST: stmopa  za3.s, { z12.b, z13.b }, z8.b, z23[3]
// CHECK-ENCODING: [0xb3,0x8d,0x48,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 80488db3 <unknown>

stmopa  za3.s, {z30.b-z31.b}, z31.b, z31[3]  // 10000000-01011111-10011111-11110011
// CHECK-INST: stmopa  za3.s, { z30.b, z31.b }, z31.b, z31[3]
// CHECK-ENCODING: [0xf3,0x9f,0x5f,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 805f9ff3 <unknown>

stmopa  za0.s, {z0.h-z1.h}, z0.h, z20[0]  // 10000000-01000000-10000000-00001000
// CHECK-INST: stmopa  za0.s, { z0.h, z1.h }, z0.h, z20[0]
// CHECK-ENCODING: [0x08,0x80,0x40,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 80408008 <unknown>

stmopa  za3.s, {z12.h-z13.h}, z8.h, z23[3]  // 10000000-01001000-10001101-10111011
// CHECK-INST: stmopa  za3.s, { z12.h, z13.h }, z8.h, z23[3]
// CHECK-ENCODING: [0xbb,0x8d,0x48,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 80488dbb <unknown>

stmopa  za3.s, {z30.h-z31.h}, z31.h, z31[3]  // 10000000-01011111-10011111-11111011
// CHECK-INST: stmopa  za3.s, { z30.h, z31.h }, z31.h, z31[3]
// CHECK-ENCODING: [0xfb,0x9f,0x5f,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 805f9ffb <unknown>
