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

fcvtx   z0.s, p0/z, z0.d  // 01100100-00011010-11000000-00000000
// CHECK-INST: fcvtx   z0.s, p0/z, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x1a,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 641ac000 <unknown>

fcvtx   z23.s, p3/z, z13.d  // 01100100-00011010-11001101-10110111
// CHECK-INST: fcvtx   z23.s, p3/z, z13.d
// CHECK-ENCODING: [0xb7,0xcd,0x1a,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 641acdb7 <unknown>

fcvtx   z31.s, p7/z, z31.d  // 01100100-00011010-11011111-11111111
// CHECK-INST: fcvtx   z31.s, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xdf,0x1a,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 641adfff <unknown>
