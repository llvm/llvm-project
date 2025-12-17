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

ustmopa za0.s, {z0.b-z1.b}, z0.b, z20[0]  // 10000001-01000000-10000000-00000000
// CHECK-INST: ustmopa za0.s, { z0.b, z1.b }, z0.b, z20[0]
// CHECK-ENCODING: [0x00,0x80,0x40,0x81]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 81408000 <unknown>

ustmopa za3.s, {z12.b-z13.b}, z8.b, z23[3]  // 10000001-01001000-10001101-10110011
// CHECK-INST: ustmopa za3.s, { z12.b, z13.b }, z8.b, z23[3]
// CHECK-ENCODING: [0xb3,0x8d,0x48,0x81]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 81488db3 <unknown>

ustmopa za3.s, {z30.b-z31.b}, z31.b, z31[3]  // 10000001-01011111-10011111-11110011
// CHECK-INST: ustmopa za3.s, { z30.b, z31.b }, z31.b, z31[3]
// CHECK-ENCODING: [0xf3,0x9f,0x5f,0x81]
// CHECK-ERROR: instruction requires: sme-tmop
// CHECK-UNKNOWN: 815f9ff3 <unknown>
