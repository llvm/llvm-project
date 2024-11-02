// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

frecpe   z0.h, z31.h
// CHECK-INST: frecpe	z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x33,0x4e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 654e33e0 <unknown>

frecpe   z0.s, z31.s
// CHECK-INST: frecpe	z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x33,0x8e,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 658e33e0 <unknown>

frecpe   z0.d, z31.d
// CHECK-INST: frecpe	z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x33,0xce,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65ce33e0 <unknown>
