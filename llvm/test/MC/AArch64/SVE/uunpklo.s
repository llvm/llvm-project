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

uunpklo z31.h, z31.b
// CHECK-INST: uunpklo	z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3b,0x72,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 3b 72 05 <unknown>

uunpklo z31.s, z31.h
// CHECK-INST: uunpklo	z31.s, z31.h
// CHECK-ENCODING: [0xff,0x3b,0xb2,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 3b b2 05 <unknown>

uunpklo z31.d, z31.s
// CHECK-INST: uunpklo	z31.d, z31.s
// CHECK-ENCODING: [0xff,0x3b,0xf2,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 3b f2 05 <unknown>
