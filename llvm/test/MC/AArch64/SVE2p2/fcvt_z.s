// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// convert to half

fcvt    z0.h, p0/z, z0.s  // 01100100-10011010-10000000-00000000
// CHECK-INST: fcvt    z0.h, p0/z, z0.s
// CHECK-ENCODING: [0x00,0x80,0x9a,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 649a8000 <unknown>

fcvt    z23.h, p3/z, z13.d  // 01100100-11011010-10001101-10110111
// CHECK-INST: fcvt    z23.h, p3/z, z13.d
// CHECK-ENCODING: [0xb7,0x8d,0xda,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64da8db7 <unknown>

// convert to single

fcvt    z0.s, p0/z, z0.h  // 01100100-10011010-10100000-00000000
// CHECK-INST: fcvt    z0.s, p0/z, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x9a,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 649aa000 <unknown>

fcvt    z31.s, p7/z, z31.d  // 01100100-11011010-11011111-11111111
// CHECK-INST: fcvt    z31.s, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xdf,0xda,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dadfff <unknown>

// convert to double

fcvt    z21.d, p5/z, z10.h  // 01100100-11011010-10110101-01010101
// CHECK-INST: fcvt    z21.d, p5/z, z10.h
// CHECK-ENCODING: [0x55,0xb5,0xda,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64dab555 <unknown>

fcvt    z31.d, p7/z, z31.s  // 01100100-11011010-11111111-11111111
// CHECK-INST: fcvt    z31.d, p7/z, z31.s
// CHECK-ENCODING: [0xff,0xff,0xda,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 64daffff <unknown