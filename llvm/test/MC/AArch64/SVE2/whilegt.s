// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

whilegt  p15.b, xzr, x0
// CHECK-INST: whilegt	p15.b, xzr, x0
// CHECK-ENCODING: [0xff,0x13,0x20,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 13 20 25 <unknown>

whilegt  p15.b, x0, xzr
// CHECK-INST: whilegt	p15.b, x0, xzr
// CHECK-ENCODING: [0x1f,0x10,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 10 3f 25 <unknown>

whilegt  p15.b, wzr, w0
// CHECK-INST: whilegt	p15.b, wzr, w0
// CHECK-ENCODING: [0xff,0x03,0x20,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 03 20 25 <unknown>

whilegt  p15.b, w0, wzr
// CHECK-INST: whilegt	p15.b, w0, wzr
// CHECK-ENCODING: [0x1f,0x00,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 00 3f 25 <unknown>

whilegt  p15.h, x0, xzr
// CHECK-INST: whilegt	p15.h, x0, xzr
// CHECK-ENCODING: [0x1f,0x10,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 10 7f 25 <unknown>

whilegt  p15.h, w0, wzr
// CHECK-INST: whilegt	p15.h, w0, wzr
// CHECK-ENCODING: [0x1f,0x00,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 00 7f 25 <unknown>

whilegt  p15.s, x0, xzr
// CHECK-INST: whilegt	p15.s, x0, xzr
// CHECK-ENCODING: [0x1f,0x10,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 10 bf 25 <unknown>

whilegt  p15.s, w0, wzr
// CHECK-INST: whilegt	p15.s, w0, wzr
// CHECK-ENCODING: [0x1f,0x00,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 00 bf 25 <unknown>

whilegt  p15.d, w0, wzr
// CHECK-INST: whilegt	p15.d, w0, wzr
// CHECK-ENCODING: [0x1f,0x00,0xff,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 00 ff 25 <unknown>

whilegt  p15.d, x0, xzr
// CHECK-INST: whilegt	p15.d, x0, xzr
// CHECK-ENCODING: [0x1f,0x10,0xff,0x25]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 1f 10 ff 25 <unknown>
