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
sdot    z23.s, z13.h, z0.h[1]  // 01000100-10001000-11001001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: sdot    z23.s, z13.h, z0.h[1]
// CHECK-ENCODING: [0xb7,0xc9,0x88,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4488c9b7 <unknown>

sdot    z0.s, z0.h, z0.h[0]  // 01000100-10000000-11001000-00000000
// CHECK-INST: sdot    z0.s, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0xc8,0x80,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4480c800 <unknown>

sdot    z21.s, z10.h, z5.h[2]  // 01000100-10010101-11001001-01010101
// CHECK-INST: sdot    z21.s, z10.h, z5.h[2]
// CHECK-ENCODING: [0x55,0xc9,0x95,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4495c955 <unknown>

sdot    z23.s, z13.h, z0.h[1]  // 01000100-10001000-11001001-10110111
// CHECK-INST: sdot    z23.s, z13.h, z0.h[1]
// CHECK-ENCODING: [0xb7,0xc9,0x88,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4488c9b7 <unknown>

sdot    z31.s, z31.h, z7.h[3]  // 01000100-10011111-11001011-11111111
// CHECK-INST: sdot    z31.s, z31.h, z7.h[3]
// CHECK-ENCODING: [0xff,0xcb,0x9f,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 449fcbff <unknown>

movprfx z23, z31
sdot    z23.s, z13.h, z8.h  // 01000100-00001000-11001001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: sdot    z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xc9,0x08,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4408c9b7 <unknown>

sdot    z0.s, z0.h, z0.h  // 01000100-00000000-11001000-00000000
// CHECK-INST: sdot    z0.s, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xc8,0x00,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4400c800 <unknown>

sdot    z21.s, z10.h, z21.h  // 01000100-00010101-11001001-01010101
// CHECK-INST: sdot    z21.s, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xc9,0x15,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4415c955 <unknown>

sdot    z23.s, z13.h, z8.h  // 01000100-00001000-11001001-10110111
// CHECK-INST: sdot    z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xc9,0x08,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4408c9b7 <unknown>

sdot    z31.s, z31.h, z31.h  // 01000100-00011111-11001011-11111111
// CHECK-INST: sdot    z31.s, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xcb,0x1f,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 441fcbff <unknown>
