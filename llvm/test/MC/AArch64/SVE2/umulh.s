// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

umulh z0.b, z1.b, z2.b
// CHECK-INST: umulh z0.b, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x6c,0x22,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 6c 22 04 <unknown>

umulh z0.h, z1.h, z2.h
// CHECK-INST: umulh z0.h, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x6c,0x62,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 6c 62 04 <unknown>

umulh z29.s, z30.s, z31.s
// CHECK-INST: umulh z29.s, z30.s, z31.s
// CHECK-ENCODING: [0xdd,0x6f,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 6f bf 04 <unknown>

umulh z31.d, z31.d, z31.d
// CHECK-INST: umulh z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x6f,0xff,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 6f ff 04 <unknown>
