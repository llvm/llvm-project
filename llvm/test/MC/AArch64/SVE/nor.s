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

nor     p0.b, p0/z, p0.b, p0.b
// CHECK-INST: nor     p0.b, p0/z, p0.b, p0.b
// CHECK-ENCODING: [0x00,0x42,0x80,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25804200 <unknown>

nor     p15.b, p15/z, p15.b, p15.b
// CHECK-INST: nor     p15.b, p15/z, p15.b, p15.b
// CHECK-ENCODING: [0xef,0x7f,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 258f7fef <unknown>
