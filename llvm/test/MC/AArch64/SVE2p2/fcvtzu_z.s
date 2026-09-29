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

fcvtzu  z0.h, p0/z, z0.h  // 01100100-01011110-11100000-00000000
// CHECK-INST: fcvtzu  z0.h, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xe0,0x5e,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645ee000 <unknown>

fcvtzu  z21.s, p5/z, z10.h  // 01100100-01011111-10110101-01010101
// CHECK-INST: fcvtzu  z21.s, p5/z, z10.h
// CHECK-ENCODING: [0x55,0xb5,0x5f,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645fb555 <unknown>

fcvtzu  z23.d, p3/z, z13.h  // 01100100-01011111-11101101-10110111
// CHECK-INST: fcvtzu  z23.d, p3/z, z13.h
// CHECK-ENCODING: [0xb7,0xed,0x5f,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 645fedb7 <unknown>

// convert from single

fcvtzu  z21.s, p5/z, z10.s  // 01100100-10011111-10110101-01010101
// CHECK-INST: fcvtzu  z21.s, p5/z, z10.s
// CHECK-ENCODING: [0x55,0xb5,0x9f,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 649fb555 <unknown>

fcvtzu  z31.d, p7/z, z31.s  // 01100100-11011111-10111111-11111111
// CHECK-INST: fcvtzu  z31.d, p7/z, z31.s
// CHECK-ENCODING: [0xff,0xbf,0xdf,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dfbfff <unknown>

// convert from double

fcvtzu  z0.s, p0/z, z0.d  // 01100100-11011110-10100000-00000000
// CHECK-INST: fcvtzu  z0.s, p0/z, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xde,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dea000 <unknown>

fcvtzu  z31.d, p7/z, z31.d  // 01100100-11011111-11111111-11111111
// CHECK-INST: fcvtzu  z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xff,0xdf,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dfffff <unknown>