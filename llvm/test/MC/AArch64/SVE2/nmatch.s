// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

nmatch p0.b, p0/z, z0.b, z0.b
// CHECK-INST: nmatch p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x10,0x80,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 10 80 20 45 <unknown>

nmatch p0.h, p0/z, z0.h, z0.h
// CHECK-INST: nmatch p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x10,0x80,0x60,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 10 80 60 45 <unknown>

nmatch p15.b, p7/z, z30.b, z31.b
// CHECK-INST: nmatch p15.b, p7/z, z30.b, z31.b
// CHECK-ENCODING: [0xdf,0x9f,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: df 9f 3f 45 <unknown>

nmatch p15.h, p7/z, z30.h, z31.h
// CHECK-INST: nmatch p15.h, p7/z, z30.h, z31.h
// CHECK-ENCODING: [0xdf,0x9f,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: df 9f 7f 45 <unknown>
