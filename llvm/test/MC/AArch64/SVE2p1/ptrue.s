// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

ptrue   pn8.h  // 00100101-01100000-01111000-00010000
// CHECK-INST: ptrue   pn8.h
// CHECK-ENCODING: [0x10,0x78,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25607810 <unknown>

ptrue   pn13.h  // 00100101-01100000-01111000-00010101
// CHECK-INST: ptrue   pn13.h
// CHECK-ENCODING: [0x15,0x78,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25607815 <unknown>

ptrue   pn15.h  // 00100101-01100000-01111000-00010111
// CHECK-INST: ptrue   pn15.h
// CHECK-ENCODING: [0x17,0x78,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25607817 <unknown>

ptrue   pn9.h  // 00100101-01100000-01111000-00010001
// CHECK-INST: ptrue   pn9.h
// CHECK-ENCODING: [0x11,0x78,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25607811 <unknown>

ptrue   pn8.s  // 00100101-10100000-01111000-00010000
// CHECK-INST: ptrue   pn8.s
// CHECK-ENCODING: [0x10,0x78,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a07810 <unknown>

ptrue   pn13.s  // 00100101-10100000-01111000-00010101
// CHECK-INST: ptrue   pn13.s
// CHECK-ENCODING: [0x15,0x78,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a07815 <unknown>

ptrue   pn15.s  // 00100101-10100000-01111000-00010111
// CHECK-INST: ptrue   pn15.s
// CHECK-ENCODING: [0x17,0x78,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a07817 <unknown>

ptrue   pn9.s  // 00100101-10100000-01111000-00010001
// CHECK-INST: ptrue   pn9.s
// CHECK-ENCODING: [0x11,0x78,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a07811 <unknown>

ptrue   pn8.d  // 00100101-11100000-01111000-00010000
// CHECK-INST: ptrue   pn8.d
// CHECK-ENCODING: [0x10,0x78,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e07810 <unknown>

ptrue   pn13.d  // 00100101-11100000-01111000-00010101
// CHECK-INST: ptrue   pn13.d
// CHECK-ENCODING: [0x15,0x78,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e07815 <unknown>

ptrue   pn15.d  // 00100101-11100000-01111000-00010111
// CHECK-INST: ptrue   pn15.d
// CHECK-ENCODING: [0x17,0x78,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e07817 <unknown>

ptrue   pn9.d  // 00100101-11100000-01111000-00010001
// CHECK-INST: ptrue   pn9.d
// CHECK-ENCODING: [0x11,0x78,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e07811 <unknown>

ptrue   pn8.b  // 00100101-00100000-01111000-00010000
// CHECK-INST: ptrue   pn8.b
// CHECK-ENCODING: [0x10,0x78,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25207810 <unknown>

ptrue   pn13.b  // 00100101-00100000-01111000-00010101
// CHECK-INST: ptrue   pn13.b
// CHECK-ENCODING: [0x15,0x78,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25207815 <unknown>

ptrue   pn15.b  // 00100101-00100000-01111000-00010111
// CHECK-INST: ptrue   pn15.b
// CHECK-ENCODING: [0x17,0x78,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25207817 <unknown>

ptrue   pn9.b  // 00100101-00100000-01111000-00010001
// CHECK-INST: ptrue   pn9.b
// CHECK-ENCODING: [0x11,0x78,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25207811 <unknown>
