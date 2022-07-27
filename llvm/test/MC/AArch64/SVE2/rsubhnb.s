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


rsubhnb z0.b, z1.h, z31.h
// CHECK-INST: rsubhnb	z0.b, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x78,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 457f7820 <unknown>

rsubhnb z0.h, z1.s, z31.s
// CHECK-INST: rsubhnb	z0.h, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x78,0xbf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45bf7820 <unknown>

rsubhnb z0.s, z1.d, z31.d
// CHECK-INST: rsubhnb	z0.s, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x78,0xff,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45ff7820 <unknown>
