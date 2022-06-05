// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

pfirst p0.b, p15, p0.b
// CHECK-INST: pfirst	p0.b, p15, p0.b
// CHECK-ENCODING: [0xe0,0xc1,0x58,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 c1 58 25 <unknown>

pfirst p15.b, p15, p15.b
// CHECK-INST: pfirst	p15.b, p15, p15.b
// CHECK-ENCODING: [0xef,0xc1,0x58,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ef c1 58 25 <unknown>
