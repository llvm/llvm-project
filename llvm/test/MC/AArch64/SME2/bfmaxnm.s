// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --mattr=+sme2,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfmaxnm {z0.h, z1.h}, {z0.h, z1.h}, z0.h  // 11000001-00100000-10100001-00100000
// CHECK-INST: bfmaxnm { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x20,0xa1,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c120a120 <unknown>

bfmaxnm {z20.h, z21.h}, {z20.h, z21.h}, z5.h  // 11000001-00100101-10100001-00110100
// CHECK-INST: bfmaxnm { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x34,0xa1,0x25,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c125a134 <unknown>

bfmaxnm {z22.h, z23.h}, {z22.h, z23.h}, z8.h  // 11000001-00101000-10100001-00110110
// CHECK-INST: bfmaxnm { z22.h, z23.h }, { z22.h, z23.h }, z8.h
// CHECK-ENCODING: [0x36,0xa1,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c128a136 <unknown>

bfmaxnm {z30.h, z31.h}, {z30.h, z31.h}, z15.h  // 11000001-00101111-10100001-00111110
// CHECK-INST: bfmaxnm { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x3e,0xa1,0x2f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c12fa13e <unknown>

bfmaxnm {z0.h, z1.h}, {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-00100000-10110001-00100000
// CHECK-INST: bfmaxnm { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x20,0xb1,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c120b120 <unknown>

bfmaxnm {z20.h, z21.h}, {z20.h, z21.h}, {z20.h, z21.h}  // 11000001-00110100-10110001-00110100
// CHECK-INST: bfmaxnm { z20.h, z21.h }, { z20.h, z21.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x34,0xb1,0x34,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c134b134 <unknown>

bfmaxnm {z22.h, z23.h}, {z22.h, z23.h}, {z8.h, z9.h}  // 11000001-00101000-10110001-00110110
// CHECK-INST: bfmaxnm { z22.h, z23.h }, { z22.h, z23.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x36,0xb1,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c128b136 <unknown>

bfmaxnm {z30.h, z31.h}, {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-00111110-10110001-00111110
// CHECK-INST: bfmaxnm { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x3e,0xb1,0x3e,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c13eb13e <unknown>

bfmaxnm {z0.h - z3.h}, {z0.h - z3.h}, z0.h  // 11000001-00100000-10101001-00100000
// CHECK-INST: bfmaxnm { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x20,0xa9,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c120a920 <unknown>

bfmaxnm {z20.h - z23.h}, {z20.h - z23.h}, z5.h  // 11000001-00100101-10101001-00110100
// CHECK-INST: bfmaxnm { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x34,0xa9,0x25,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c125a934 <unknown>

bfmaxnm {z20.h - z23.h}, {z20.h - z23.h}, z8.h  // 11000001-00101000-10101001-00110100
// CHECK-INST: bfmaxnm { z20.h - z23.h }, { z20.h - z23.h }, z8.h
// CHECK-ENCODING: [0x34,0xa9,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c128a934 <unknown>

bfmaxnm {z28.h - z31.h}, {z28.h - z31.h}, z15.h  // 11000001-00101111-10101001-00111100
// CHECK-INST: bfmaxnm { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x3c,0xa9,0x2f,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c12fa93c <unknown>

bfmaxnm {z0.h - z3.h}, {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-00100000-10111001-00100000
// CHECK-INST: bfmaxnm { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x20,0xb9,0x20,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c120b920 <unknown>

bfmaxnm {z20.h - z23.h}, {z20.h - z23.h}, {z20.h - z23.h}  // 11000001-00110100-10111001-00110100
// CHECK-INST: bfmaxnm { z20.h - z23.h }, { z20.h - z23.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x34,0xb9,0x34,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c134b934 <unknown>

bfmaxnm {z20.h - z23.h}, {z20.h - z23.h}, {z8.h - z11.h}  // 11000001-00101000-10111001-00110100
// CHECK-INST: bfmaxnm { z20.h - z23.h }, { z20.h - z23.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x34,0xb9,0x28,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c128b934 <unknown>

bfmaxnm {z28.h - z31.h}, {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-00111100-10111001-00111100
// CHECK-INST: bfmaxnm { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x3c,0xb9,0x3c,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2
// CHECK-UNKNOWN: c13cb93c <unknown>
