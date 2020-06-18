// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

frecps z0.h, z1.h, z31.h
// CHECK-INST: frecps	z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x18,0x5f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 18 5f 65 <unknown>

frecps z0.s, z1.s, z31.s
// CHECK-INST: frecps	z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x18,0x9f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 18 9f 65 <unknown>

frecps z0.d, z1.d, z31.d
// CHECK-INST: frecps	z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x18,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 18 df 65 <unknown>
