// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

uunpkhi z31.h, z31.b
// CHECK-INST: uunpkhi	z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3b,0x73,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b 73 05 <unknown>

uunpkhi z31.s, z31.h
// CHECK-INST: uunpkhi	z31.s, z31.h
// CHECK-ENCODING: [0xff,0x3b,0xb3,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b b3 05 <unknown>

uunpkhi z31.d, z31.s
// CHECK-INST: uunpkhi	z31.d, z31.s
// CHECK-ENCODING: [0xff,0x3b,0xf3,0x05]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3b f3 05 <unknown>
