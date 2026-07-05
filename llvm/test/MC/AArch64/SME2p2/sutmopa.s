// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-tmop < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-tmop < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-tmop --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-tmop < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-tmop --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-tmop < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-tmop -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sutmopa za0.s, {z0.b-z1.b}, z0.b, z20[0]  // 10000000-01100000-10000000-00000000
// CHECK-INST: sutmopa za0.s, { z0.b, z1.b }, z0.b, z20[0]
// CHECK-ENCODING: [0x00,0x80,0x60,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 80608000 <unknown>

sutmopa za1.s, {z10.b-z11.b}, z21.b, z29[1]  // 10000000-01110101-10010101-01010001
// CHECK-INST: sutmopa za1.s, { z10.b, z11.b }, z21.b, z29[1]
// CHECK-ENCODING: [0x51,0x95,0x75,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 80759551 <unknown>

sutmopa za3.s, {z30.b-z31.b}, z31.b, z31[3]  // 10000000-01111111-10011111-11110011
// CHECK-INST: sutmopa za3.s, { z30.b, z31.b }, z31.b, z31[3]
// CHECK-ENCODING: [0xf3,0x9f,0x7f,0x80]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 807f9ff3 <unknown>
