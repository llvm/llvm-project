// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ftssel    z0.h, z1.h, z31.h
// CHECK-INST: ftssel	z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0xb0,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 b0 7f 04 <unknown>

ftssel    z0.s, z1.s, z31.s
// CHECK-INST: ftssel	z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0xb0,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 b0 bf 04 <unknown>

ftssel    z0.d, z1.d, z31.d
// CHECK-INST: ftssel	z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0xb0,0xff,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 20 b0 ff 04 <unknown>
