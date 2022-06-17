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

ld1rsw  { z0.d }, p0/z, [x0]
// CHECK-INST: ld1rsw  { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x80,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 c0 84 <unknown>

ld1rsw  { z31.d }, p7/z, [sp, #252]
// CHECK-INST: ld1rsw  { z31.d }, p7/z, [sp, #252]
// CHECK-ENCODING: [0xff,0x9f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f ff 84 <unknown>
