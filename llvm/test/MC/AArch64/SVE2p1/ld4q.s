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

ld4q    {z0.q, z1.q, z2.q, z3.q}, p0/z, [x0, x0, lsl #4]  // 10100101-10100000-10000000-00000000
// CHECK-INST: ld4q    { z0.q - z3.q }, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a5a08000 <unknown>

ld4q    {z21.q, z22.q, z23.q, z24.q}, p5/z, [x10, x21, lsl #4]  // 10100101-10110101-10010101-01010101
// CHECK-INST: ld4q    { z21.q - z24.q }, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x55,0x95,0xb5,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a5b59555 <unknown>

ld4q    {z23.q, z24.q, z25.q, z26.q}, p3/z, [x13, x8, lsl #4]  // 10100101-10101000-10001101-10110111
// CHECK-INST: ld4q    { z23.q - z26.q }, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xb7,0x8d,0xa8,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a5a88db7 <unknown>

ld4q    {z0.q, z1.q, z2.q, z3.q}, p0/z, [x0]  // 10100101-10010000-11100000-00000000
// CHECK-INST: ld4q    { z0.q - z3.q }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x90,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a590e000 <unknown>

ld4q    {z21.q, z22.q, z23.q, z24.q}, p5/z, [x10, #20, mul vl]  // 10100101-10010101-11110101-01010101
// CHECK-INST: ld4q    { z21.q - z24.q }, p5/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x95,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a595f555 <unknown>

ld4q    {z23.q, z24.q, z25.q, z26.q}, p3/z, [x13, #-32, mul vl]  // 10100101-10011000-11101101-10110111
// CHECK-INST: ld4q    { z23.q - z26.q }, p3/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x98,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a598edb7 <unknown>

ld4q    {z31.q, z0.q, z1.q, z2.q}, p7/z, [sp, #-4, mul vl]  // 10100101-10011111-11111111-11111111
// CHECK-INST: ld4q    { z31.q, z0.q, z1.q, z2.q }, p7/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x9f,0xa5]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a59fffff <unknown>
