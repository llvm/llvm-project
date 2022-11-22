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

ld3q    {z0.q, z1.q, z2.q}, p0/z, [x0, x0, lsl #4]  // 10100101-00100000-10000000-00000000
// CHECK-INST: ld3q    { z0.q - z2.q }, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0x20,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a5208000 <unknown>

ld3q    {z21.q, z22.q, z23.q}, p5/z, [x10, x21, lsl #4]  // 10100101-00110101-10010101-01010101
// CHECK-INST: ld3q    { z21.q - z23.q }, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x55,0x95,0x35,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a5359555 <unknown>

ld3q    {z23.q, z24.q, z25.q}, p3/z, [x13, x8, lsl #4]  // 10100101-00101000-10001101-10110111
// CHECK-INST: ld3q    { z23.q - z25.q }, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xb7,0x8d,0x28,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a5288db7 <unknown>

ld3q    {z0.q, z1.q, z2.q}, p0/z, [x0]  // 10100101-00010000-11100000-00000000
// CHECK-INST: ld3q    { z0.q - z2.q }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x10,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a510e000 <unknown>

ld3q    {z21.q, z22.q, z23.q}, p5/z, [x10, #15, mul vl]  // 10100101-00010101-11110101-01010101
// CHECK-INST: ld3q    { z21.q - z23.q }, p5/z, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x15,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a515f555 <unknown>

ld3q    {z23.q, z24.q, z25.q}, p3/z, [x13, #-24, mul vl]  // 10100101-00011000-11101101-10110111
// CHECK-INST: ld3q    { z23.q - z25.q }, p3/z, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x18,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a518edb7 <unknown>

ld3q    {z31.q, z0.q, z1.q}, p7/z, [sp, #-3, mul vl]  // 10100101-00011111-11111111-11111111
// CHECK-INST: ld3q    { z31.q, z0.q, z1.q }, p7/z, [sp, #-3, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a51fffff <unknown>
