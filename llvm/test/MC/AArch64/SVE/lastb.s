// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

lastb   w0, p7, z31.b
// CHECK-INST: lastb	w0, p7, z31.b
// CHECK-ENCODING: [0xe0,0xbf,0x21,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0521bfe0 <unknown>

lastb   w0, p7, z31.h
// CHECK-INST: lastb	w0, p7, z31.h
// CHECK-ENCODING: [0xe0,0xbf,0x61,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0561bfe0 <unknown>

lastb   w0, p7, z31.s
// CHECK-INST: lastb	w0, p7, z31.s
// CHECK-ENCODING: [0xe0,0xbf,0xa1,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a1bfe0 <unknown>

lastb   x0, p7, z31.d
// CHECK-INST: lastb	x0, p7, z31.d
// CHECK-ENCODING: [0xe0,0xbf,0xe1,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e1bfe0 <unknown>

lastb   b0, p7, z31.b
// CHECK-INST: lastb	b0, p7, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x23,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05239fe0 <unknown>

lastb   h0, p7, z31.h
// CHECK-INST: lastb	h0, p7, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x63,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05639fe0 <unknown>

lastb   s0, p7, z31.s
// CHECK-INST: lastb	s0, p7, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xa3,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a39fe0 <unknown>

lastb   d0, p7, z31.d
// CHECK-INST: lastb	d0, p7, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe3,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e39fe0 <unknown>
