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
bfmlslt z23.s, z13.h, z8.h  // 01100100-11101000-10100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmlslt z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xa5,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e8a5b7 <unknown>

bfmlslt z0.s, z0.h, z0.h  // 01100100-11100000-10100100-00000000
// CHECK-INST: bfmlslt z0.s, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xa4,0xe0,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e0a400 <unknown>

bfmlslt z21.s, z10.h, z21.h  // 01100100-11110101-10100101-01010101
// CHECK-INST: bfmlslt z21.s, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xa5,0xf5,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64f5a555 <unknown>

bfmlslt z23.s, z13.h, z8.h  // 01100100-11101000-10100101-10110111
// CHECK-INST: bfmlslt z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xa5,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e8a5b7 <unknown>

bfmlslt z31.s, z31.h, z31.h  // 01100100-11111111-10100111-11111111
// CHECK-INST: bfmlslt z31.s, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xa7,0xff,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64ffa7ff <unknown>

movprfx z23, z31
bfmlslt z23.s, z13.h, z0.h[3]  // 01100100-11101000-01101101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmlslt z23.s, z13.h, z0.h[3]
// CHECK-ENCODING: [0xb7,0x6d,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e86db7 <unknown>

bfmlslt z0.s, z0.h, z0.h[0]  // 01100100-11100000-01100100-00000000
// CHECK-INST: bfmlslt z0.s, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x64,0xe0,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e06400 <unknown>

bfmlslt z21.s, z10.h, z5.h[4]  // 01100100-11110101-01100101-01010101
// CHECK-INST: bfmlslt z21.s, z10.h, z5.h[4]
// CHECK-ENCODING: [0x55,0x65,0xf5,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64f56555 <unknown>

bfmlslt z23.s, z13.h, z0.h[3]  // 01100100-11101000-01101101-10110111
// CHECK-INST: bfmlslt z23.s, z13.h, z0.h[3]
// CHECK-ENCODING: [0xb7,0x6d,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e86db7 <unknown>

bfmlslt z31.s, z31.h, z7.h[7]  // 01100100-11111111-01101111-11111111
// CHECK-INST: bfmlslt z31.s, z31.h, z7.h[7]
// CHECK-ENCODING: [0xff,0x6f,0xff,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64ff6fff <unknown>
