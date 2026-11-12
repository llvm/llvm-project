// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+f8f16mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+f8f16mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2,+f8f16mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+f8f16mm < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+f8f16mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+f8f16mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movprfx z23, z31
fmmla   z23.h, z13.b, z8.b  // 01100100-01101000-11100001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmmla   z23.h, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0xe1,0x68,0x64]
// CHECK-ERROR: instruction requires: f8f16mm sve2
// CHECK-UNKNOWN: 6468e1b7 <unknown>

fmmla   z0.h, z0.b, z0.b  // 01100100-01100000-11100000-00000000
// CHECK-INST: fmmla   z0.h, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xe0,0x60,0x64]
// CHECK-ERROR: instruction requires: f8f16mm sve2
// CHECK-UNKNOWN: 6460e000 <unknown>

fmmla   z21.h, z10.b, z21.b  // 01100100-01110101-11100001-01010101
// CHECK-INST: fmmla   z21.h, z10.b, z21.b
// CHECK-ENCODING: [0x55,0xe1,0x75,0x64]
// CHECK-ERROR: instruction requires: f8f16mm sve2
// CHECK-UNKNOWN: 6475e155 <unknown>

fmmla   z31.h, z31.b, z31.b  // 01100100-01111111-11100011-11111111
// CHECK-INST: fmmla   z31.h, z31.b, z31.b
// CHECK-ENCODING: [0xff,0xe3,0x7f,0x64]
// CHECK-ERROR: instruction requires: f8f16mm sve2
// CHECK-UNKNOWN: 647fe3ff <unknown>