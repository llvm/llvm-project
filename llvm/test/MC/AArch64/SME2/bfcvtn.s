// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


bfcvtn  z0.h, {z0.s, z1.s}  // 11000001-01100000-11100000-00100000
// CHECK-INST: bfcvtn  z0.h, { z0.s, z1.s }
// CHECK-ENCODING: [0x20,0xe0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e020 <unknown>

bfcvtn  z21.h, {z10.s, z11.s}  // 11000001-01100000-11100001-01110101
// CHECK-INST: bfcvtn  z21.h, { z10.s, z11.s }
// CHECK-ENCODING: [0x75,0xe1,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e175 <unknown>

bfcvtn  z23.h, {z12.s, z13.s}  // 11000001-01100000-11100001-10110111
// CHECK-INST: bfcvtn  z23.h, { z12.s, z13.s }
// CHECK-ENCODING: [0xb7,0xe1,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e1b7 <unknown>

bfcvtn  z31.h, {z30.s, z31.s}  // 11000001-01100000-11100011-11111111
// CHECK-INST: bfcvtn  z31.h, { z30.s, z31.s }
// CHECK-ENCODING: [0xff,0xe3,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160e3ff <unknown>

