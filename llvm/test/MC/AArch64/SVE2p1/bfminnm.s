// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sve2,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-b16b16 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-b16b16 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movprfx  z23.h, p3/m, z31.h
bfminnm z23.h, p3/m, z23.h, z13.h  // 01100101-00000101-10001101-10110111
// CHECK-INST:  movprfx  z23.h, p3/m, z31.h
// CHECK-INST: bfminnm z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x05,0x65]
// CHECK-ERROR: instruction requires: b16b16 sve2 or sme2
// CHECK-UNKNOWN: 65058db7 <unknown>

movprfx z23, z31
bfminnm z23.h, p3/m, z23.h, z13.h  // 01100101-00000101-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfminnm z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x05,0x65]
// CHECK-ERROR: instruction requires: b16b16 sve2 or sme2
// CHECK-UNKNOWN: 65058db7 <unknown>

bfminnm z0.h, p0/m, z0.h, z0.h  // 01100101-00000101-10000000-00000000
// CHECK-INST: bfminnm z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x05,0x65]
// CHECK-ERROR: instruction requires: b16b16 sve2 or sme2
// CHECK-UNKNOWN: 65058000 <unknown>

bfminnm z21.h, p5/m, z21.h, z10.h  // 01100101-00000101-10010101-01010101
// CHECK-INST: bfminnm z21.h, p5/m, z21.h, z10.h
// CHECK-ENCODING: [0x55,0x95,0x05,0x65]
// CHECK-ERROR: instruction requires: b16b16 sve2 or sme2
// CHECK-UNKNOWN: 65059555 <unknown>

bfminnm z23.h, p3/m, z23.h, z13.h  // 01100101-00000101-10001101-10110111
// CHECK-INST: bfminnm z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x05,0x65]
// CHECK-ERROR: instruction requires: b16b16 sve2 or sme2
// CHECK-UNKNOWN: 65058db7 <unknown>

bfminnm z31.h, p7/m, z31.h, z31.h  // 01100101-00000101-10011111-11111111
// CHECK-INST: bfminnm z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x9f,0x05,0x65]
// CHECK-ERROR: instruction requires: b16b16 sve2 or sme2
// CHECK-UNKNOWN: 65059fff <unknown>

