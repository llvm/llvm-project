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

sel     p0.b, p0, p0.b, p0.b
// CHECK-INST: mov     p0.b, p0/m, p0.b
// CHECK-ENCODING: [0x10,0x42,0x00,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25004210 <unknown>

sel     p15.b, p15, p15.b, p15.b
// CHECK-INST: mov     p15.b, p15/m, p15.b
// CHECK-ENCODING: [0xff,0x7f,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 250f7fff <unknown>

sel     z31.b, p15, z31.b, z31.b
// CHECK-INST: mov     z31.b, p15/m, z31.b
// CHECK-ENCODING: [0xff,0xff,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 053fffff <unknown>

sel     z31.h, p15, z31.h, z31.h
// CHECK-INST: mov     z31.h, p15/m, z31.h
// CHECK-ENCODING: [0xff,0xff,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 057fffff <unknown>

sel     z31.s, p15, z31.s, z31.s
// CHECK-INST: mov     z31.s, p15/m, z31.s
// CHECK-ENCODING: [0xff,0xff,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05bfffff <unknown>

sel     z31.d, p15, z31.d, z31.d
// CHECK-INST: mov     z31.d, p15/m, z31.d
// CHECK-ENCODING: [0xff,0xff,0xff,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05ffffff <unknown>

sel     z23.s, p11, z13.s, z8.s
// CHECK-INST: sel     z23.s, p11, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0xed,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a8edb7 <unknown>

sel     z23.d, p11, z13.d, z8.d
// CHECK-INST: sel     z23.d, p11, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0xed,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e8edb7 <unknown>

sel     z23.h, p11, z13.h, z8.h
// CHECK-INST: sel     z23.h, p11, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xed,0x68,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0568edb7 <unknown>

sel     z23.b, p11, z13.b, z8.b
// CHECK-INST: sel     z23.b, p11, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xed,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0528edb7 <unknown>
