// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cls     z31.b, p7/m, z31.b
// CHECK-INST: cls	z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x18,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 18 04 <unknown>

cls     z31.h, p7/m, z31.h
// CHECK-INST: cls	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x58,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 58 04 <unknown>

cls     z31.s, p7/m, z31.s
// CHECK-INST: cls	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x98,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 98 04 <unknown>

cls     z31.d, p7/m, z31.d
// CHECK-INST: cls	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xd8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf d8 04 <unknown>
