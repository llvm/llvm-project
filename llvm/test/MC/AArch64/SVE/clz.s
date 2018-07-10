// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

clz     z31.b, p7/m, z31.b
// CHECK-INST: clz	z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x19,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 19 04 <unknown>

clz     z31.h, p7/m, z31.h
// CHECK-INST: clz	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x59,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 59 04 <unknown>

clz     z31.s, p7/m, z31.s
// CHECK-INST: clz	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x99,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 99 04 <unknown>

clz     z31.d, p7/m, z31.d
// CHECK-INST: clz	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xd9,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf d9 04 <unknown>
