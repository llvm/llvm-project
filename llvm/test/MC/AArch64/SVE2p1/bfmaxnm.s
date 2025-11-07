// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+sve-b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s 2>&1 \
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

movprfx  z23.h, p3/m, z31.h
bfmaxnm z23.h, p3/m, z23.h, z13.h  // 01100101-00000100-10001101-10110111
// CHECK-INST:  movprfx  z23.h, p3/m, z31.h
// CHECK-INST: bfmaxnm z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x04,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65048db7 <unknown>

movprfx z23, z31
bfmaxnm z23.h, p3/m, z23.h, z13.h  // 01100101-00000100-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmaxnm z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x04,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65048db7 <unknown>

bfmaxnm z0.h, p0/m, z0.h, z0.h  // 01100101-00000100-10000000-00000000
// CHECK-INST: bfmaxnm z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x04,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65048000 <unknown>

bfmaxnm z21.h, p5/m, z21.h, z10.h  // 01100101-00000100-10010101-01010101
// CHECK-INST: bfmaxnm z21.h, p5/m, z21.h, z10.h
// CHECK-ENCODING: [0x55,0x95,0x04,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65049555 <unknown>

bfmaxnm z23.h, p3/m, z23.h, z13.h  // 01100101-00000100-10001101-10110111
// CHECK-INST: bfmaxnm z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x04,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65048db7 <unknown>

bfmaxnm z31.h, p7/m, z31.h, z31.h  // 01100101-00000100-10011111-11111111
// CHECK-INST: bfmaxnm z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x9f,0x04,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65049fff <unknown>

