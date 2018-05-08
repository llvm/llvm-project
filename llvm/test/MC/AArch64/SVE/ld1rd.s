// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1rd   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1rd   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 c0 85 <unknown>

ld1rd   { z31.d }, p7/z, [sp, #504]
// CHECK-INST: ld1rd   { z31.d }, p7/z, [sp, #504]
// CHECK-ENCODING: [0xff,0xff,0xff,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff ff 85 <unknown>
