// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN


subhnb z0.b, z1.h, z31.h
// CHECK-INST: subhnb	z0.b, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x70,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 70 7f 45 <unknown>

subhnb z0.h, z1.s, z31.s
// CHECK-INST: subhnb	z0.h, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x70,0xbf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 70 bf 45 <unknown>

subhnb z0.s, z1.d, z31.d
// CHECK-INST: subhnb	z0.s, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x70,0xff,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 70 ff 45 <unknown>
