// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cnt     z31.b, p7/m, z31.b
// CHECK-INST: cnt	z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x1a,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 1a 04 <unknown>

cnt     z31.h, p7/m, z31.h
// CHECK-INST: cnt	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x5a,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 5a 04 <unknown>

cnt     z31.s, p7/m, z31.s
// CHECK-INST: cnt	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x9a,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 9a 04 <unknown>

cnt     z31.d, p7/m, z31.d
// CHECK-INST: cnt	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xda,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf da 04 <unknown>
