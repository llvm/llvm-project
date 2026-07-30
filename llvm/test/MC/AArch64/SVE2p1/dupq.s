// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


dupq    z0.h, z0.h[0]  // 00000101-00100010-00100100-00000000
// CHECK-INST: dupq    z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x24,0x22,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05222400 <unknown>

dupq    z21.h, z10.h[5]  // 00000101-00110110-00100101-01010101
// CHECK-INST: dupq    z21.h, z10.h[5]
// CHECK-ENCODING: [0x55,0x25,0x36,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05362555 <unknown>

dupq    z23.h, z13.h[2]  // 00000101-00101010-00100101-10110111
// CHECK-INST: dupq    z23.h, z13.h[2]
// CHECK-ENCODING: [0xb7,0x25,0x2a,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052a25b7 <unknown>

dupq    z31.h, z31.h[7]  // 00000101-00111110-00100111-11111111
// CHECK-INST: dupq    z31.h, z31.h[7]
// CHECK-ENCODING: [0xff,0x27,0x3e,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 053e27ff <unknown>


dupq    z0.s, z0.s[0]  // 00000101-00100100-00100100-00000000
// CHECK-INST: dupq    z0.s, z0.s[0]
// CHECK-ENCODING: [0x00,0x24,0x24,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05242400 <unknown>

dupq    z21.s, z10.s[2]  // 00000101-00110100-00100101-01010101
// CHECK-INST: dupq    z21.s, z10.s[2]
// CHECK-ENCODING: [0x55,0x25,0x34,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05342555 <unknown>

dupq    z23.s, z13.s[1]  // 00000101-00101100-00100101-10110111
// CHECK-INST: dupq    z23.s, z13.s[1]
// CHECK-ENCODING: [0xb7,0x25,0x2c,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052c25b7 <unknown>

dupq    z31.s, z31.s[3]  // 00000101-00111100-00100111-11111111
// CHECK-INST: dupq    z31.s, z31.s[3]
// CHECK-ENCODING: [0xff,0x27,0x3c,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 053c27ff <unknown>


dupq    z0.d, z0.d[0]  // 00000101-00101000-00100100-00000000
// CHECK-INST: dupq    z0.d, z0.d[0]
// CHECK-ENCODING: [0x00,0x24,0x28,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05282400 <unknown>

dupq    z21.d, z10.d[1]  // 00000101-00111000-00100101-01010101
// CHECK-INST: dupq    z21.d, z10.d[1]
// CHECK-ENCODING: [0x55,0x25,0x38,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05382555 <unknown>

dupq    z23.d, z13.d[0]  // 00000101-00101000-00100101-10110111
// CHECK-INST: dupq    z23.d, z13.d[0]
// CHECK-ENCODING: [0xb7,0x25,0x28,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052825b7 <unknown>

dupq    z31.d, z31.d[1]  // 00000101-00111000-00100111-11111111
// CHECK-INST: dupq    z31.d, z31.d[1]
// CHECK-ENCODING: [0xff,0x27,0x38,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 053827ff <unknown>


dupq    z0.b, z0.b[0]  // 00000101-00100001-00100100-00000000
// CHECK-INST: dupq    z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x24,0x21,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05212400 <unknown>

dupq    z21.b, z10.b[10]  // 00000101-00110101-00100101-01010101
// CHECK-INST: dupq    z21.b, z10.b[10]
// CHECK-ENCODING: [0x55,0x25,0x35,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05352555 <unknown>

dupq    z23.b, z13.b[4]  // 00000101-00101001-00100101-10110111
// CHECK-INST: dupq    z23.b, z13.b[4]
// CHECK-ENCODING: [0xb7,0x25,0x29,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052925b7 <unknown>

dupq    z31.b, z31.b[15]  // 00000101-00111111-00100111-11111111
// CHECK-INST: dupq    z31.b, z31.b[15]
// CHECK-ENCODING: [0xff,0x27,0x3f,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 053f27ff <unknown>

