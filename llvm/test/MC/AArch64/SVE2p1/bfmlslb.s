// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movprfx z23, z31
bfmlslb z23.s, z13.h, z8.h  // 01100100-11101000-10100001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmlslb z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xa1,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e8a1b7 <unknown>

bfmlslb z0.s, z0.h, z0.h  // 01100100-11100000-10100000-00000000
// CHECK-INST: bfmlslb z0.s, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xa0,0xe0,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e0a000 <unknown>

bfmlslb z21.s, z10.h, z21.h  // 01100100-11110101-10100001-01010101
// CHECK-INST: bfmlslb z21.s, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xa1,0xf5,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64f5a155 <unknown>

bfmlslb z23.s, z13.h, z8.h  // 01100100-11101000-10100001-10110111
// CHECK-INST: bfmlslb z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xa1,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e8a1b7 <unknown>

bfmlslb z31.s, z31.h, z31.h  // 01100100-11111111-10100011-11111111
// CHECK-INST: bfmlslb z31.s, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xa3,0xff,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64ffa3ff <unknown>

movprfx z23, z31
bfmlslb z23.s, z13.h, z0.h[3]  // 01100100-11101000-01101001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmlslb z23.s, z13.h, z0.h[3]
// CHECK-ENCODING: [0xb7,0x69,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e869b7 <unknown>

bfmlslb z0.s, z0.h, z0.h[0]  // 01100100-11100000-01100000-00000000
// CHECK-INST: bfmlslb z0.s, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x60,0xe0,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e06000 <unknown>

bfmlslb z21.s, z10.h, z5.h[4]  // 01100100-11110101-01100001-01010101
// CHECK-INST: bfmlslb z21.s, z10.h, z5.h[4]
// CHECK-ENCODING: [0x55,0x61,0xf5,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64f56155 <unknown>

bfmlslb z23.s, z13.h, z0.h[3]  // 01100100-11101000-01101001-10110111
// CHECK-INST: bfmlslb z23.s, z13.h, z0.h[3]
// CHECK-ENCODING: [0xb7,0x69,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e869b7 <unknown>

bfmlslb z31.s, z31.h, z7.h[7]  // 01100100-11111111-01101011-11111111
// CHECK-INST: bfmlslb z31.s, z31.h, z7.h[7]
// CHECK-ENCODING: [0xff,0x6b,0xff,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64ff6bff <unknown>
