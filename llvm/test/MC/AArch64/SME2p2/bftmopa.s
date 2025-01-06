// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-b16b16 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// non-widening

bftmopa za0.h, {z0.h-z1.h}, z0.h, z20[0]  // 10000001-01100000-00000000-00001000
// CHECK-INST: bftmopa za0.h, { z0.h, z1.h }, z0.h, z20[0]
// CHECK-ENCODING: [0x08,0x00,0x60,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81600008 <unknown>

bftmopa za1.h, {z12.h-z13.h}, z8.h, z23[3]  // 10000001-01101000-00001101-10111001
// CHECK-INST: bftmopa za1.h, { z12.h, z13.h }, z8.h, z23[3]
// CHECK-ENCODING: [0xb9,0x0d,0x68,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 81680db9 <unknown>

bftmopa za1.h, {z30.h-z31.h}, z31.h, z31[3]  // 10000001-01111111-00011111-11111001
// CHECK-INST: bftmopa za1.h, { z30.h, z31.h }, z31.h, z31[3]
// CHECK-ENCODING: [0xf9,0x1f,0x7f,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-b16b16
// CHECK-UNKNOWN: 817f1ff9 <unknown>

// widening

bftmopa za0.s, {z0.h-z1.h}, z0.h, z20[0]  // 10000001-01000000-00000000-00000000
// CHECK-INST: bftmopa za0.s, { z0.h, z1.h }, z0.h, z20[0]
// CHECK-ENCODING: [0x00,0x00,0x40,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81400000 <unknown>

bftmopa za3.s, {z12.h-z13.h}, z8.h, z23[3]  // 10000001-01001000-00001101-10110011
// CHECK-INST: bftmopa za3.s, { z12.h, z13.h }, z8.h, z23[3]
// CHECK-ENCODING: [0xb3,0x0d,0x48,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81480db3 <unknown>

bftmopa za3.s, {z30.h-z31.h}, z31.h, z31[3]  // 10000001-01011111-00011111-11110011
// CHECK-INST: bftmopa za3.s, { z30.h, z31.h }, z31.h, z31[3]
// CHECK-ENCODING: [0xf3,0x1f,0x5f,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 815f1ff3 <unknown>
