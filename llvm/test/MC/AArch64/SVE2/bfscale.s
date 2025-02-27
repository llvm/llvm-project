// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-bfscale < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-bfscale < %s \
// RUN:        | llvm-objdump -d --mattr=+sve,+sve-bfscale - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-bfscale < %s \
// RUN:        | llvm-objdump -d --mattr=-sve-bfscale - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-bfscale < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve,+sve-bfscale -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfscale z0.h, p0/m, z0.h, z0.h  // 01100101-00001001-10000000-00000000
// CHECK-INST: bfscale z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x09,0x65]
// CHECK-ERROR: instruction requires: sve-bfscale
// CHECK-UNKNOWN: 65098000 <unknown>

bfscale z21.h, p5/m, z21.h, z10.h  // 01100101-00001001-10010101-01010101
// CHECK-INST: bfscale z21.h, p5/m, z21.h, z10.h
// CHECK-ENCODING: [0x55,0x95,0x09,0x65]
// CHECK-ERROR: instruction requires: sve-bfscale
// CHECK-UNKNOWN: 65099555 <unknown>

bfscale z31.h, p7/m, z31.h, z31.h  // 01100101-00001001-10011111-11111111
// CHECK-INST: bfscale z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x9f,0x09,0x65]
// CHECK-ERROR: instruction requires: sve-bfscale
// CHECK-UNKNOWN: 65099fff <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx  z23.h, p3/m, z31.h
bfscale z23.h, p3/m, z23.h, z13.h  // 01100101-00001001-10001101-10110111
// CHECK-INST:  movprfx  z23.h, p3/m, z31.h
// CHECK-INST: bfscale z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x09,0x65]
// CHECK-ERROR: instruction requires: sve-bfscale
// CHECK-UNKNOWN: 65098db7 <unknown>

movprfx z23, z31
bfscale z23.h, p3/m, z23.h, z13.h  // 01100101-00001001-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfscale z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x09,0x65]
// CHECK-ERROR: instruction requires: sve-bfscale
// CHECK-UNKNOWN: 65098db7 <unknown>
