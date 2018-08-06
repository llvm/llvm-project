// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fneg    z31.h, p7/m, z31.h
// CHECK-INST: fneg	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x5d,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 5d 04 <unknown>

fneg    z31.s, p7/m, z31.s
// CHECK-INST: fneg	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x9d,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 9d 04 <unknown>

fneg    z31.d, p7/m, z31.d
// CHECK-INST: fneg	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xdd,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf dd 04 <unknown>
