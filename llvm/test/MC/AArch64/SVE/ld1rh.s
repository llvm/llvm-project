// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1rh   { z0.h }, p0/z, [x0]
// CHECK-INST: ld1rh   { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 c0 84 <unknown>

ld1rh   { z0.s }, p0/z, [x0]
// CHECK-INST: ld1rh   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xc0,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 c0 84 <unknown>

ld1rh   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1rh   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 c0 84 <unknown>

ld1rh   { z31.h }, p7/z, [sp, #126]
// CHECK-INST: ld1rh   { z31.h }, p7/z, [sp, #126]
// CHECK-ENCODING: [0xff,0xbf,0xff,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf ff 84 <unknown>

ld1rh   { z31.s }, p7/z, [sp, #126]
// CHECK-INST: ld1rh   { z31.s }, p7/z, [sp, #126]
// CHECK-ENCODING: [0xff,0xdf,0xff,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df ff 84 <unknown>

ld1rh   { z31.d }, p7/z, [sp, #126]
// CHECK-INST: ld1rh   { z31.d }, p7/z, [sp, #126]
// CHECK-ENCODING: [0xff,0xff,0xff,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff ff 84 <unknown>
