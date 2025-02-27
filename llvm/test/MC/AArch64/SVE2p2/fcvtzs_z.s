// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// convert from half

fcvtzs  z0.h, p0/z, z0.h  // 01100100-01011110-11000000-00000000
// CHECK-INST: fcvtzs  z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xc0,0x5e,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645ec000 <unknown>

fcvtzs  z23.s, p3/z, z13.h  // 01100100-01011111-10001101-10110111
// CHECK-INST: fcvtzs  z23.s, p3/z, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x5f,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645f8db7 <unknown>

fcvtzs  z31.d, p7/z, z31.h  // 01100100-01011111-11011111-11111111
// CHECK-INST: fcvtzs  z31.d, p7/z, z31.h
// CHECK-ENCODING: [0xff,0xdf,0x5f,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645fdfff <unknown>

// convert from single

fcvtzs  z0.s, p0/z, z0.s  // 01100100-10011111-10000000-00000000
// CHECK-INST: fcvtzs  z0.s, p0/z, z0.s
// CHECK-ENCODING: [0x00,0x80,0x9f,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 649f8000 <unknown>

fcvtzs  z21.d, p5/z, z10.s  // 01100100-11011111-10010101-01010101
// CHECK-INST: fcvtzs  z21.d, p5/z, z10.s
// CHECK-ENCODING: [0x55,0x95,0xdf,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64df9555 <unknown>

// convert from double

fcvtzs  z23.s, p3/z, z13.d  // 01100100-11011110-10001101-10110111
// CHECK-INST: fcvtzs  z23.s, p3/z, z13.d
// CHECK-ENCODING: [0xb7,0x8d,0xde,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64de8db7 <unknown>

fcvtzs  z31.d, p7/z, z31.d  // 01100100-11011111-11011111-11111111
// CHECK-INST: fcvtzs  z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xdf,0xdf,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dfdfff <unknown>
