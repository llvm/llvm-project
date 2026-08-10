// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sve-b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sve-b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfclamp {z0.h, z1.h}, z0.h, z0.h  // 11000001-00100000-11000000-00000000
// CHECK-INST: bfclamp { z0.h, z1.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc0,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c120c000 <unknown>

bfclamp {z20.h, z21.h}, z10.h, z21.h  // 11000001-00110101-11000001-01010100
// CHECK-INST: bfclamp { z20.h, z21.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xc1,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c135c154 <unknown>

bfclamp {z22.h, z23.h}, z13.h, z8.h  // 11000001-00101000-11000001-10110110
// CHECK-INST: bfclamp { z22.h, z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb6,0xc1,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c128c1b6 <unknown>

bfclamp {z30.h, z31.h}, z31.h, z31.h  // 11000001-00111111-11000011-11111110
// CHECK-INST: bfclamp { z30.h, z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfe,0xc3,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c13fc3fe <unknown>

bfclamp {z0.h - z3.h}, z0.h, z0.h  // 11000001-00100000-11001000-00000000
// CHECK-INST: bfclamp { z0.h - z3.h }, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc8,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c120c800 <unknown>

bfclamp {z20.h - z23.h}, z10.h, z21.h  // 11000001-00110101-11001001-01010100
// CHECK-INST: bfclamp { z20.h - z23.h }, z10.h, z21.h
// CHECK-ENCODING: [0x54,0xc9,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c135c954 <unknown>

bfclamp {z20.h - z23.h}, z13.h, z8.h  // 11000001-00101000-11001001-10110100
// CHECK-INST: bfclamp { z20.h - z23.h }, z13.h, z8.h
// CHECK-ENCODING: [0xb4,0xc9,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c128c9b4 <unknown>

bfclamp {z28.h - z31.h}, z31.h, z31.h  // 11000001-00111111-11001011-11111100
// CHECK-INST: bfclamp { z28.h - z31.h }, z31.h, z31.h
// CHECK-ENCODING: [0xfc,0xcb,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2 sve-b16b16
// CHECK-UNKNOWN: c13fcbfc <unknown>
