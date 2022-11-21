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
fdot    z23.s, z13.h, z8.h  // 01100100-00101000-10000001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fdot    z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x81,0x28,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 642881b7 <unknown>

fdot    z0.s, z0.h, z0.h  // 01100100-00100000-10000000-00000000
// CHECK-INST: fdot    z0.s, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x20,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64208000 <unknown>

fdot    z21.s, z10.h, z21.h  // 01100100-00110101-10000001-01010101
// CHECK-INST: fdot    z21.s, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x81,0x35,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64358155 <unknown>

fdot    z23.s, z13.h, z8.h  // 01100100-00101000-10000001-10110111
// CHECK-INST: fdot    z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x81,0x28,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 642881b7 <unknown>

fdot    z31.s, z31.h, z31.h  // 01100100-00111111-10000011-11111111
// CHECK-INST: fdot    z31.s, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x83,0x3f,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 643f83ff <unknown>

movprfx z23, z31
fdot    z23.s, z13.h, z0.h[1]  // 01100100-00101000-01000001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fdot    z23.s, z13.h, z0.h[1]
// CHECK-ENCODING: [0xb7,0x41,0x28,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 642841b7 <unknown>

fdot    z0.s, z0.h, z0.h[0]  // 01100100-00100000-01000000-00000000
// CHECK-INST: fdot    z0.s, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x40,0x20,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64204000 <unknown>

fdot    z21.s, z10.h, z5.h[2]  // 01100100-00110101-01000001-01010101
// CHECK-INST: fdot    z21.s, z10.h, z5.h[2]
// CHECK-ENCODING: [0x55,0x41,0x35,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64354155 <unknown>

fdot    z23.s, z13.h, z0.h[1]  // 01100100-00101000-01000001-10110111
// CHECK-INST: fdot    z23.s, z13.h, z0.h[1]
// CHECK-ENCODING: [0xb7,0x41,0x28,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 642841b7 <unknown>

fdot    z31.s, z31.h, z7.h[3]  // 01100100-00111111-01000011-11111111
// CHECK-INST: fdot    z31.s, z31.h, z7.h[3]
// CHECK-ENCODING: [0xff,0x43,0x3f,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 643f43ff <unknown>
