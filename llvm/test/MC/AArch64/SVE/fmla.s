// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmla z0.h, p7/m, z1.h, z31.h
// CHECK-INST: fmla	z0.h, p7/m, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x1c,0x7f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 1c 7f 65 <unknown>

fmla z0.s, p7/m, z1.s, z31.s
// CHECK-INST: fmla	z0.s, p7/m, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x1c,0xbf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 1c bf 65 <unknown>

fmla z0.d, p7/m, z1.d, z31.d
// CHECK-INST: fmla	z0.d, p7/m, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x1c,0xff,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 1c ff 65 <unknown>

fmla z0.h, z1.h, z7.h[7]
// CHECK-INST: fmla	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x00,0x7f,0x64]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 00 7f 64 <unknown>

fmla z0.s, z1.s, z7.s[3]
// CHECK-INST: fmla	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0x00,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 00 bf 64 <unknown>

fmla z0.d, z1.d, z7.d[1]
// CHECK-INST: fmla	z0.d, z1.d, z7.d[1]
// CHECK-ENCODING: [0x20,0x00,0xf7,0x64]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 00 f7 64 <unknown>
