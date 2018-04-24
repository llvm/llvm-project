// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldff1sw { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1sw { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 9f a4 <unknown>

ldff1sw { z31.d }, p7/z, [sp, xzr, lsl #2]
// CHECK-INST: ldff1sw { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 9f a4 <unknown>

ldff1sw { z0.d }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ldff1sw { z0.d }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x60,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 80 a4 <unknown>
