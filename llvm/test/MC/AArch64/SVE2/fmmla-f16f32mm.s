
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-f16f32mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-f16f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve,+sve-f16f32mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-f16f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-f16f32mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve,+sve-f16f32mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


movprfx z23, z31
fmmla   z23.s, z13.h, z8.h  // 01100100-00101000-11100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fmmla   z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xe5,0x28,0x64]
// CHECK-ERROR: instruction requires: sve-f16f32mm
// CHECK-UNKNOWN: 6428e5b7 <unknown>

fmmla   z0.s, z0.h, z0.h  // 01100100-00100000-11100100-00000000
// CHECK-INST: fmmla   z0.s, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xe4,0x20,0x64]
// CHECK-ERROR: instruction requires: sve-f16f32mm
// CHECK-UNKNOWN: 6420e400 <unknown>

fmmla   z23.s, z13.h, z8.h  // 01100100-00101000-11100101-10110111
// CHECK-INST: fmmla   z23.s, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0xe5,0x28,0x64]
// CHECK-ERROR: instruction requires: sve-f16f32mm
// CHECK-UNKNOWN: 6428e5b7 <unknown>

fmmla   z31.s, z31.h, z31.h  // 01100100-00111111-11100111-11111111
// CHECK-INST: fmmla   z31.s, z31.h, z31.h
// CHECK-ENCODING: [0xff,0xe7,0x3f,0x64]
// CHECK-ERROR: instruction requires: sve-f16f32mm
// CHECK-UNKNOWN: 643fe7ff <unknown>