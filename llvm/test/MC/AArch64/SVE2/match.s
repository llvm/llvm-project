// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

match p0.b, p0/z, z0.b, z0.b
// CHECK-INST: match p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 45208000 <unknown>

match p0.h, p0/z, z0.h, z0.h
// CHECK-INST: match p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x60,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 45608000 <unknown>

match p15.b, p7/z, z30.b, z31.b
// CHECK-INST: match p15.b, p7/z, z30.b, z31.b
// CHECK-ENCODING: [0xcf,0x9f,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 453f9fcf <unknown>

match p15.h, p7/z, z30.h, z31.h
// CHECK-INST: match p15.h, p7/z, z30.h, z31.h
// CHECK-ENCODING: [0xcf,0x9f,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 457f9fcf <unknown>
