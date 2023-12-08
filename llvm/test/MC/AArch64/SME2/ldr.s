// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


ldr     zt0, [x0]  // 11100001-00011111-10000000-00000000
// CHECK-INST: ldr     zt0, [x0]
// CHECK-ENCODING: [0x00,0x80,0x1f,0xe1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: e11f8000 <unknown>

ldr     zt0, [x10]  // 11100001-00011111-10000001-01000000
// CHECK-INST: ldr     zt0, [x10]
// CHECK-ENCODING: [0x40,0x81,0x1f,0xe1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: e11f8140 <unknown>

ldr     zt0, [x13]  // 11100001-00011111-10000001-10100000
// CHECK-INST: ldr     zt0, [x13]
// CHECK-ENCODING: [0xa0,0x81,0x1f,0xe1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: e11f81a0 <unknown>

ldr     zt0, [sp]  // 11100001-00011111-10000011-11100000
// CHECK-INST: ldr     zt0, [sp]
// CHECK-ENCODING: [0xe0,0x83,0x1f,0xe1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: e11f83e0 <unknown>

