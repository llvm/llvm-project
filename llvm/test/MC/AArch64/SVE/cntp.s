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

cntp  x0, p15, p0.b
// CHECK-INST: cntp	x0, p15, p0.b
// CHECK-ENCODING: [0x00,0xbc,0x20,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2520bc00 <unknown>

cntp  x0, p15, p0.h
// CHECK-INST: cntp	x0, p15, p0.h
// CHECK-ENCODING: [0x00,0xbc,0x60,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2560bc00 <unknown>

cntp  x0, p15, p0.s
// CHECK-INST: cntp	x0, p15, p0.s
// CHECK-ENCODING: [0x00,0xbc,0xa0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a0bc00 <unknown>

cntp  x0, p15, p0.d
// CHECK-INST: cntp	x0, p15, p0.d
// CHECK-ENCODING: [0x00,0xbc,0xe0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e0bc00 <unknown>
