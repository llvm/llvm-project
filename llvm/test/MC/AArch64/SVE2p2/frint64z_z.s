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

frint64z z0.d, p0/z, z0.d  // 01100100-00011101-11000000-00000000
// CHECK-INST: frint64z z0.d, p0/z, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x1d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 641dc000 <unknown>

frint64z z21.s, p5/z, z10.s  // 01100100-00011101-10010101-01010101
// CHECK-INST: frint64z z21.s, p5/z, z10.s
// CHECK-ENCODING: [0x55,0x95,0x1d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 641d9555 <unknown>

frint64z z31.s, p7/z, z31.s  // 01100100-00011101-10011111-11111111
// CHECK-INST: frint64z z31.s, p7/z, z31.s
// CHECK-ENCODING: [0xff,0x9f,0x1d,0x64]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 641d9fff <unknown>