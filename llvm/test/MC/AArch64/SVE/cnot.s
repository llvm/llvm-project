// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cnot    z31.b, p7/m, z31.b
// CHECK-INST: cnot	z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x1b,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 1b 04 <unknown>

cnot    z31.h, p7/m, z31.h
// CHECK-INST: cnot	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x5b,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 5b 04 <unknown>

cnot    z31.s, p7/m, z31.s
// CHECK-INST: cnot	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 9b 04 <unknown>

cnot    z31.d, p7/m, z31.d
// CHECK-INST: cnot	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xdb,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf db 04 <unknown>
