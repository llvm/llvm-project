// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fexpa z0.h, z31.h
// CHECK-INST: fexpa	z0.h, z31.h
// CHECK-ENCODING: [0xe0,0xbb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme2p2
// CHECK-UNKNOWN: 0460bbe0 <unknown>

fexpa z0.s, z31.s
// CHECK-INST: fexpa	z0.s, z31.s
// CHECK-ENCODING: [0xe0,0xbb,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme2p2
// CHECK-UNKNOWN: 04a0bbe0 <unknown>

fexpa z0.d, z31.d
// CHECK-INST: fexpa	z0.d, z31.d
// CHECK-ENCODING: [0xe0,0xbb,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme2p2
// CHECK-UNKNOWN: 04e0bbe0 <unknown>
