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
udot    z23.s, z13.h, z0.h[1]  // 01000100-10001000-11001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: udot    z23.s, z13.h, z0.h[1]
// CHECK-ENCODING: [0xb7,0xcd,0x88,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4488cdb7 <unknown>

udot    z0.s, z0.h, z0.h[0]  // 01000100-10000000-11001100-00000000
// CHECK-INST: udot    z0.s, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0xcc,0x80,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4480cc00 <unknown>

udot    z21.s, z10.h, z5.h[2]  // 01000100-10010101-11001101-01010101
// CHECK-INST: udot    z21.s, z10.h, z5.h[2]
// CHECK-ENCODING: [0x55,0xcd,0x95,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4495cd55 <unknown>

udot    z23.s, z13.h, z0.h[1]  // 01000100-10001000-11001101-10110111
// CHECK-INST: udot    z23.s, z13.h, z0.h[1]
// CHECK-ENCODING: [0xb7,0xcd,0x88,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4488cdb7 <unknown>

udot    z31.s, z31.h, z7.h[3]  // 01000100-10011111-11001111-11111111
// CHECK-INST: udot    z31.s, z31.h, z7.h[3]
// CHECK-ENCODING: [0xff,0xcf,0x9f,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 449fcfff <unknown>

movprfx z23, z31
udot    z23.s, z13.h, z8.h  // 01000100-00001000-11001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: udot    z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xcd,0x08,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4408cdb7 <unknown>

udot    z0.s, z0.h, z0.h  // 01000100-00000000-11001100-00000000
// CHECK-INST: udot    z0.s, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xcc,0x00,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4400cc00 <unknown>

udot    z21.s, z10.h, z21.h  // 01000100-00010101-11001101-01010101
// CHECK-INST: udot    z21.s, z10.h, z21.h
// CHECK-ENCODING: [0x55,0xcd,0x15,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4415cd55 <unknown>

udot    z23.s, z13.h, z8.h  // 01000100-00001000-11001101-10110111
// CHECK-INST: udot    z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xcd,0x08,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 4408cdb7 <unknown>

udot    z31.s, z31.h, z31.h  // 01000100-00011111-11001111-11111111
// CHECK-INST: udot    z31.s, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xcf,0x1f,0x44]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 441fcfff <unknown>
