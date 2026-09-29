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

movprfx  z23.d, p3/m, z31.d
frint64z z23.d, p3/m, z13.d  // 01100101-00010110-10101101-10110111
// CHECK-INST:  movprfx  z23.d, p3/m, z31.d
// CHECK-INST: frint64z z23.d, p3/m, z13.d
// CHECK-ENCODING: [0xb7,0xad,0x16,0x65]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6516adb7 <unknown>

movprfx z23, z31
frint64z z23.s, p3/m, z13.s  // 01100101-00010100-10101101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: frint64z z23.s, p3/m, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x14,0x65]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6514adb7 <unknown>

frint64z z0.s, p0/m, z0.s  // 01100101-00010100-10100000-00000000
// CHECK-INST: frint64z z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x14,0x65]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6514a000 <unknown>

frint64z z21.d, p5/m, z10.d  // 01100101-00010110-10110101-01010101
// CHECK-INST: frint64z z21.d, p5/m, z10.d
// CHECK-ENCODING: [0x55,0xb5,0x16,0x65]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6516b555 <unknown>

frint64z z31.d, p7/m, z31.d  // 01100101-00010110-10111111-11111111
// CHECK-INST: frint64z z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0x16,0x65]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 6516bfff <unknown>