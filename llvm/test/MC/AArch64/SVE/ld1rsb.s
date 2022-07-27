// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1rsb  { z0.h }, p0/z, [x0]
// CHECK-INST: ld1rsb  { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xc0,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0c000 <unknown>

ld1rsb  { z0.s }, p0/z, [x0]
// CHECK-INST: ld1rsb  { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0a000 <unknown>

ld1rsb  { z0.d }, p0/z, [x0]
// CHECK-INST: ld1rsb  { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x80,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c08000 <unknown>

ld1rsb  { z31.h }, p7/z, [sp, #63]
// CHECK-INST: ld1rsb  { z31.h }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0xdf,0xff,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85ffdfff <unknown>

ld1rsb  { z31.s }, p7/z, [sp, #63]
// CHECK-INST: ld1rsb  { z31.s }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0xbf,0xff,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85ffbfff <unknown>

ld1rsb  { z31.d }, p7/z, [sp, #63]
// CHECK-INST: ld1rsb  { z31.d }, p7/z, [sp, #63]
// CHECK-ENCODING: [0xff,0x9f,0xff,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85ff9fff <unknown>
