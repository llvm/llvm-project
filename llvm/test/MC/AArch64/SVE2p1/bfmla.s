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

movprfx z23, z31
bfmla   z23.h, z13.h, z0.h[5]  // 01100100-01101000-00001001-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmla   z23.h, z13.h, z0.h[5]
// CHECK-ENCODING: [0xb7,0x09,0x68,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 646809b7 <unknown>

bfmla   z0.h, z0.h, z0.h[0]  // 01100100-00100000-00001000-00000000
// CHECK-INST: bfmla   z0.h, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x08,0x20,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 64200800 <unknown>

bfmla   z21.h, z10.h, z5.h[6]  // 01100100-01110101-00001001-01010101
// CHECK-INST: bfmla   z21.h, z10.h, z5.h[6]
// CHECK-ENCODING: [0x55,0x09,0x75,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 64750955 <unknown>

bfmla   z23.h, z13.h, z0.h[5]  // 01100100-01101000-00001001-10110111
// CHECK-INST: bfmla   z23.h, z13.h, z0.h[5]
// CHECK-ENCODING: [0xb7,0x09,0x68,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 646809b7 <unknown>

bfmla   z31.h, z31.h, z7.h[7]  // 01100100-01111111-00001011-11111111
// CHECK-INST: bfmla   z31.h, z31.h, z7.h[7]
// CHECK-ENCODING: [0xff,0x0b,0x7f,0x64]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 647f0bff <unknown>


movprfx  z23.h, p3/m, z31.h
bfmla   z23.h, p3/m, z13.h, z8.h  // 01100101-00101000-00001101-10110111
// CHECK-INST:  movprfx  z23.h, p3/m, z31.h
// CHECK-INST: bfmla   z23.h, p3/m, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x0d,0x28,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65280db7 <unknown>

movprfx z23, z31
bfmla   z23.h, p3/m, z13.h, z8.h  // 01100101-00101000-00001101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: bfmla   z23.h, p3/m, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x0d,0x28,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65280db7 <unknown>

bfmla   z0.h, p0/m, z0.h, z0.h  // 01100101-00100000-00000000-00000000
// CHECK-INST: bfmla   z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x20,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65200000 <unknown>

bfmla   z21.h, p5/m, z10.h, z21.h  // 01100101-00110101-00010101-01010101
// CHECK-INST: bfmla   z21.h, p5/m, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x15,0x35,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65351555 <unknown>

bfmla   z23.h, p3/m, z13.h, z8.h  // 01100101-00101000-00001101-10110111
// CHECK-INST: bfmla   z23.h, p3/m, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x0d,0x28,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 65280db7 <unknown>

bfmla   z31.h, p7/m, z31.h, z31.h  // 01100101-00111111-00011111-11111111
// CHECK-INST: bfmla   z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x3f,0x65]
// CHECK-ERROR: instruction requires: sve-b16b16
// CHECK-UNKNOWN: 653f1fff <unknown>

