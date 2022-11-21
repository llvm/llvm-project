// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movprfx z23, z31
fclamp  z23.d, z13.d, z8.d  // 01100100-11101000-00100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fclamp  z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x25,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e825b7 <unknown>

fclamp  z0.d, z0.d, z0.d  // 01100100-11100000-00100100-00000000
// CHECK-INST: fclamp  z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x24,0xe0,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e02400 <unknown>

fclamp  z21.d, z10.d, z21.d  // 01100100-11110101-00100101-01010101
// CHECK-INST: fclamp  z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0x25,0xf5,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64f52555 <unknown>

fclamp  z23.d, z13.d, z8.d  // 01100100-11101000-00100101-10110111
// CHECK-INST: fclamp  z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x25,0xe8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64e825b7 <unknown>

fclamp  z31.d, z31.d, z31.d  // 01100100-11111111-00100111-11111111
// CHECK-INST: fclamp  z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x27,0xff,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64ff27ff <unknown>

movprfx z23, z31
fclamp  z23.h, z13.h, z8.h  // 01100100-01101000-00100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fclamp  z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x25,0x68,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 646825b7 <unknown>

fclamp  z0.h, z0.h, z0.h  // 01100100-01100000-00100100-00000000
// CHECK-INST: fclamp  z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x24,0x60,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64602400 <unknown>

fclamp  z21.h, z10.h, z21.h  // 01100100-01110101-00100101-01010101
// CHECK-INST: fclamp  z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x25,0x75,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64752555 <unknown>

fclamp  z23.h, z13.h, z8.h  // 01100100-01101000-00100101-10110111
// CHECK-INST: fclamp  z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x25,0x68,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 646825b7 <unknown>

fclamp  z31.h, z31.h, z31.h  // 01100100-01111111-00100111-11111111
// CHECK-INST: fclamp  z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x27,0x7f,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 647f27ff <unknown>

movprfx z23, z31
fclamp  z23.s, z13.s, z8.s  // 01100100-10101000-00100101-10110111
// CHECK-INST:  movprfx z23, z31
// CHECK-INST: fclamp  z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0x25,0xa8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64a825b7 <unknown>

fclamp  z0.s, z0.s, z0.s  // 01100100-10100000-00100100-00000000
// CHECK-INST: fclamp  z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x24,0xa0,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64a02400 <unknown>

fclamp  z21.s, z10.s, z21.s  // 01100100-10110101-00100101-01010101
// CHECK-INST: fclamp  z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0x25,0xb5,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64b52555 <unknown>

fclamp  z23.s, z13.s, z8.s  // 01100100-10101000-00100101-10110111
// CHECK-INST: fclamp  z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0x25,0xa8,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64a825b7 <unknown>

fclamp  z31.s, z31.s, z31.s  // 01100100-10111111-00100111-11111111
// CHECK-INST: fclamp  z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x27,0xbf,0x64]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 64bf27ff <unknown>
