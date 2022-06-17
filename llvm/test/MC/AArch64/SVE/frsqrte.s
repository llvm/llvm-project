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

frsqrte  z0.h, z31.h
// CHECK-INST: frsqrte	z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x33,0x4f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 33 4f 65 <unknown>

frsqrte  z0.s, z31.s
// CHECK-INST: frsqrte	z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x33,0x8f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 33 8f 65 <unknown>

frsqrte  z0.d, z31.d
// CHECK-INST: frsqrte	z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x33,0xcf,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 33 cf 65 <unknown>
