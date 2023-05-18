// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1sb   z0.h, p0/z, [x0]
// CHECK-INST: ld1sb   { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5c0a000 <unknown>

ld1sb   z0.s, p0/z, [x0]
// CHECK-INST: ld1sb   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5a0a000 <unknown>

ld1sb   z0.d, p0/z, [x0]
// CHECK-INST: ld1sb   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a580a000 <unknown>

ld1sb   { z0.h }, p0/z, [x0]
// CHECK-INST: ld1sb   { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5c0a000 <unknown>

ld1sb   { z0.s }, p0/z, [x0]
// CHECK-INST: ld1sb   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5a0a000 <unknown>

ld1sb   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1sb   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a580a000 <unknown>

ld1sb   { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1sb   { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xcf,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5cfbfff <unknown>

ld1sb   { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1sb   { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xc5,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5c5b555 <unknown>

ld1sb   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1sb   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xaf,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5afbfff <unknown>

ld1sb   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1sb   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xa5,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5a5b555 <unknown>

ld1sb   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1sb   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x8f,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a58fbfff <unknown>

ld1sb   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1sb   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x85,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a585b555 <unknown>

ld1sb    { z0.h }, p0/z, [sp, x0]
// CHECK-INST: ld1sb    { z0.h }, p0/z, [sp, x0]
// CHECK-ENCODING: [0xe0,0x43,0xc0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5c043e0 <unknown>

ld1sb    { z0.h }, p0/z, [x0, x0]
// CHECK-INST: ld1sb    { z0.h }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0xc0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5c04000 <unknown>

ld1sb    { z0.h }, p0/z, [x0, x0, lsl #0]
// CHECK-INST: ld1sb    { z0.h }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0xc0,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5c04000 <unknown>

ld1sb    { z21.s }, p5/z, [x10, x21]
// CHECK-INST: ld1sb    { z21.s }, p5/z, [x10, x21]
// CHECK-ENCODING: [0x55,0x55,0xb5,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5b55555 <unknown>

ld1sb    { z23.d }, p3/z, [x13, x8]
// CHECK-INST: ld1sb    { z23.d }, p3/z, [x13, x8]
// CHECK-ENCODING: [0xb7,0x4d,0x88,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a5884db7 <unknown>
