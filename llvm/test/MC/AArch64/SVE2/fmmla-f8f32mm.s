// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+f8f32mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+f8f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+f8f32mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+f8f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+f8f32mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+f8f32mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movprfx z23, z31
fmmla   z23.s, z13.b, z8.b  // 01100100-00101000-11100001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmmla   z23.s, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xe1,0x28,0x64]
// CHECK-ERROR: instruction requires: f8f32mm sve2
// CHECK-UNKNOWN: 6428e1b7 <unknown>

fmmla   z0.s, z0.b, z0.b  // 01100100-00100000-11100000-00000000
// CHECK-INST: fmmla   z0.s, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xe0,0x20,0x64]
// CHECK-ERROR: instruction requires: f8f32mm sve2
// CHECK-UNKNOWN: 6420e000 <unknown>

fmmla   z21.s, z10.b, z21.b  // 01100100-00110101-11100001-01010101
// CHECK-INST: fmmla   z21.s, z10.b, z21.b
// CHECK-ENCODING: [0x55,0xe1,0x35,0x64]
// CHECK-ERROR: instruction requires: f8f32mm sve2
// CHECK-UNKNOWN: 6435e155 <unknown>

fmmla   z31.s, z31.b, z31.b  // 01100100-00111111-11100011-11111111
// CHECK-INST: fmmla   z31.s, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xe3,0x3f,0x64]
// CHECK-ERROR: instruction requires: f8f32mm sve2
// CHECK-UNKNOWN: 643fe3ff <unknown>
