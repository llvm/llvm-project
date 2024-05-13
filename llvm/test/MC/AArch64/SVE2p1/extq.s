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


movprfx z23, z31
extq    z23.b, z23.b, z13.b, #8  // 00000101-01101000-00100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: extq    z23.b, z23.b, z13.b, #8
// CHECK-ENCODING: [0xb7,0x25,0x68,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056825b7 <unknown>

extq    z0.b, z0.b, z0.b, #0  // 00000101-01100000-00100100-00000000
// CHECK-INST: extq    z0.b, z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0x24,0x60,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05602400 <unknown>

extq    z21.b, z21.b, z10.b, #5  // 00000101-01100101-00100101-01010101
// CHECK-INST: extq    z21.b, z21.b, z10.b, #5
// CHECK-ENCODING: [0x55,0x25,0x65,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05652555 <unknown>

extq    z23.b, z23.b, z13.b, #8  // 00000101-01101000-00100101-10110111
// CHECK-INST: extq    z23.b, z23.b, z13.b, #8
// CHECK-ENCODING: [0xb7,0x25,0x68,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056825b7 <unknown>

extq    z31.b, z31.b, z31.b, #15  // 00000101-01101111-00100111-11111111
// CHECK-INST: extq    z31.b, z31.b, z31.b, #15
// CHECK-ENCODING: [0xff,0x27,0x6f,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056f27ff <unknown>

