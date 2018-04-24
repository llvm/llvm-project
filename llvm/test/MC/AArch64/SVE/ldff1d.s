// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldff1d  { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1d  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xff,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f ff a5 <unknown>

ldff1d  { z31.d }, p7/z, [sp, xzr, lsl #3]
// CHECK-INST: ldff1d  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xff,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f ff a5 <unknown>

ldff1d  { z0.d }, p0/z, [x0, x0, lsl #3]
// CHECK-INST: ldff1d  { z0.d }, p0/z, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x60,0xe0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 e0 a5 <unknown>
