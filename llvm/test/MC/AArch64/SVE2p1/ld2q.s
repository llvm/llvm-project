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

ld2q    {z0.q, z1.q}, p0/z, [x0, x0, lsl #4]  // 10100100-10100000-10000000-00000000
// CHECK-INST: ld2q    { z0.q, z1.q }, p0/z, [x0, x0, lsl #4]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a4a08000 <unknown>

ld2q    {z21.q, z22.q}, p5/z, [x10, x21, lsl #4]  // 10100100-10110101-10010101-01010101
// CHECK-INST: ld2q    { z21.q, z22.q }, p5/z, [x10, x21, lsl #4]
// CHECK-ENCODING: [0x55,0x95,0xb5,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a4b59555 <unknown>

ld2q    {z23.q, z24.q}, p3/z, [x13, x8, lsl #4]  // 10100100-10101000-10001101-10110111
// CHECK-INST: ld2q    { z23.q, z24.q }, p3/z, [x13, x8, lsl #4]
// CHECK-ENCODING: [0xb7,0x8d,0xa8,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a4a88db7 <unknown>

ld2q    {z0.q, z1.q}, p0/z, [x0]  // 10100100-10010000-11100000-00000000
// CHECK-INST: ld2q    { z0.q, z1.q }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x90,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a490e000 <unknown>

ld2q    {z21.q, z22.q}, p5/z, [x10, #10, mul vl]  // 10100100-10010101-11110101-01010101
// CHECK-INST: ld2q    { z21.q, z22.q }, p5/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x95,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a495f555 <unknown>

ld2q    {z23.q, z24.q}, p3/z, [x13, #-16, mul vl]  // 10100100-10011000-11101101-10110111
// CHECK-INST: ld2q    { z23.q, z24.q }, p3/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x98,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a498edb7 <unknown>

ld2q    {z31.q, z0.q}, p7/z, [sp, #-2, mul vl]  // 10100100-10011111-11111111-11111111
// CHECK-INST: ld2q    { z31.q, z0.q }, p7/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x9f,0xa4]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: a49fffff <unknown>
