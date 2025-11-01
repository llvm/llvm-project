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

bfmul   z0.h, z0.h, z0.h[0]  // 01100100-00100000-00101000-00000000
// CHECK-INST: bfmul   z0.h, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x28,0x20,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 64202800 <unknown>

bfmul   z21.h, z10.h, z5.h[6]  // 01100100-01110101-00101001-01010101
// CHECK-INST: bfmul   z21.h, z10.h, z5.h[6]
// CHECK-ENCODING: [0x55,0x29,0x75,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 64752955 <unknown>

bfmul   z23.h, z13.h, z0.h[5]  // 01100100-01101000-00101001-10110111
// CHECK-INST: bfmul   z23.h, z13.h, z0.h[5]
// CHECK-ENCODING: [0xb7,0x29,0x68,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 646829b7 <unknown>

bfmul   z31.h, z31.h, z7.h[7]  // 01100100-01111111-00101011-11111111
// CHECK-INST: bfmul   z31.h, z31.h, z7.h[7]
// CHECK-ENCODING: [0xff,0x2b,0x7f,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 647f2bff <unknown>

movprfx  z23.h, p3/m, z31.h
bfmul   z23.h, p3/m, z23.h, z13.h  // 01100101-00000010-10001101-10110111
// CHECK-INST:  movprfx  z23.h, p3/m, z31.h
// CHECK-INST: bfmul   z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x02,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65028db7 <unknown>

movprfx z23, z31
bfmul   z23.h, p3/m, z23.h, z13.h  // 01100101-00000010-10001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmul   z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x02,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65028db7 <unknown>

bfmul   z0.h, p0/m, z0.h, z0.h  // 01100101-00000010-10000000-00000000
// CHECK-INST: bfmul   z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x02,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65028000 <unknown>

bfmul   z21.h, p5/m, z21.h, z10.h  // 01100101-00000010-10010101-01010101
// CHECK-INST: bfmul   z21.h, p5/m, z21.h, z10.h
// CHECK-ENCODING: [0x55,0x95,0x02,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65029555 <unknown>

bfmul   z23.h, p3/m, z23.h, z13.h  // 01100101-00000010-10001101-10110111
// CHECK-INST: bfmul   z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x8d,0x02,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65028db7 <unknown>

bfmul   z31.h, p7/m, z31.h, z31.h  // 01100101-00000010-10011111-11111111
// CHECK-INST: bfmul   z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x9f,0x02,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65029fff <unknown>

bfmul   z0.h, z0.h, z0.h  // 01100101-00000000-00001000-00000000
// CHECK-INST: bfmul   z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x08,0x00,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65000800 <unknown>

bfmul   z21.h, z10.h, z21.h  // 01100101-00010101-00001001-01010101
// CHECK-INST: bfmul   z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x09,0x15,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65150955 <unknown>

bfmul   z23.h, z13.h, z8.h  // 01100101-00001000-00001001-10110111
// CHECK-INST: bfmul   z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x09,0x08,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 650809b7 <unknown>

bfmul   z31.h, z31.h, z31.h  // 01100101-00011111-00001011-11111111
// CHECK-INST: bfmul   z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x0b,0x1f,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 651f0bff <unknown>

