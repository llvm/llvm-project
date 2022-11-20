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

ld1rb   { z0.b }, p0/z, [x0]
// CHECK-INST: ld1rb   { z0.b }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x80,0x40,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 84408000 <unknown>

ld1rb   { z0.h }, p0/z, [x0]
// CHECK-INST: ld1rb   { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x40,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 8440a000 <unknown>

ld1rb   { z0.s }, p0/z, [x0]
// CHECK-INST: ld1rb   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xc0,0x40,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 8440c000 <unknown>

ld1rb   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1rb   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 8440e000 <unknown>

ld1rb   { z31.b }, p7/z, [sp, #63]
// CHECK-INST: ld1rb   { z31.b }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0x9f,0x7f,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 847f9fff <unknown>

ld1rb   { z31.h }, p7/z, [sp, #63]
// CHECK-INST: ld1rb   { z31.h }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0xbf,0x7f,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 847fbfff <unknown>

ld1rb   { z31.s }, p7/z, [sp, #63]
// CHECK-INST: ld1rb   { z31.s }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0xdf,0x7f,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 847fdfff <unknown>

ld1rb   { z31.d }, p7/z, [sp, #63]
// CHECK-INST: ld1rb   { z31.d }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0xff,0x7f,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 847fffff <unknown>
