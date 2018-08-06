// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

uxth    z0.s, p0/m, z0.s
// CHECK-INST: uxth    z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x93,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 93 04 <unknown>

uxth    z0.d, p0/m, z0.d
// CHECK-INST: uxth    z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd3,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d3 04 <unknown>

uxth    z31.s, p7/m, z31.s
// CHECK-INST: uxth    z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x93,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 93 04 <unknown>

uxth    z31.d, p7/m, z31.d
// CHECK-INST: uxth    z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xd3,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf d3 04 <unknown>
