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


raddhnb z0.b, z1.h, z31.h
// CHECK-INST: raddhnb	z0.b, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x68,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 457f6820 <unknown>

raddhnb z0.h, z1.s, z31.s
// CHECK-INST: raddhnb	z0.h, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x68,0xbf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45bf6820 <unknown>

raddhnb z0.s, z1.d, z31.d
// CHECK-INST: raddhnb	z0.s, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x68,0xff,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45ff6820 <unknown>
