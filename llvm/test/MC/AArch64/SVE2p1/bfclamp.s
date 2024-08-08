// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+sve-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+sve-b16b16 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sve2,+sve-b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+sve-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve-b16b16 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+sve-b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+sve-b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2,+sve-b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve-b16b16 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sve-b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movprfx z23, z31
bfclamp z23.h, z13.h, z8.h  // 01100100-00101000-00100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfclamp z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x25,0x28,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme2 sve-b16b16 
// CHECK-UNKNOWN: 642825b7 <unknown>

bfclamp z0.h, z0.h, z0.h  // 01100100-00100000-00100100-00000000
// CHECK-INST: bfclamp z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x24,0x20,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme2 sve-b16b16 
// CHECK-UNKNOWN: 64202400 <unknown>

bfclamp z21.h, z10.h, z21.h  // 01100100-00110101-00100101-01010101
// CHECK-INST: bfclamp z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x25,0x35,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme2 sve-b16b16 
// CHECK-UNKNOWN: 64352555 <unknown>

bfclamp z23.h, z13.h, z8.h  // 01100100-00101000-00100101-10110111
// CHECK-INST: bfclamp z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x25,0x28,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme2 sve-b16b16
// CHECK-UNKNOWN: 642825b7 <unknown>

bfclamp z31.h, z31.h, z31.h  // 01100100-00111111-00100111-11111111
// CHECK-INST: bfclamp z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x27,0x3f,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme2 sve-b16b16
// CHECK-UNKNOWN: 643f27ff <unknown>

