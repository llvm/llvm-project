// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

tbx  z31.b, z31.b, z31.b
// CHECK-INST: tbx	z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x2f,0x3f,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2f 3f 05 <unknown>

tbx  z31.h, z31.h, z31.h
// CHECK-INST: tbx	z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x2f,0x7f,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2f 7f 05 <unknown>

tbx  z31.s, z31.s, z31.s
// CHECK-INST: tbx	z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x2f,0xbf,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2f bf 05 <unknown>

tbx  z31.d, z31.d, z31.d
// CHECK-INST: tbx	z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x2f,0xff,0x05]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2f ff 05 <unknown>
