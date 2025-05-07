// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfcvtnt z0.h, p0/z, z0.s  // 01100100-10000010-10100000-00000000
// CHECK-INST: bfcvtnt z0.h, p0/z, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x82,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6482a000 <unknown>

bfcvtnt z23.h, p3/z, z13.s  // 01100100-10000010-10101101-10110111
// CHECK-INST: bfcvtnt z23.h, p3/z, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x82,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6482adb7 <unknown>

bfcvtnt z31.h, p7/z, z31.s  // 01100100-10000010-10111111-11111111
// CHECK-INST: bfcvtnt z31.h, p7/z, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x82,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6482bfff <unknown>