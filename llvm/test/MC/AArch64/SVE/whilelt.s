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

whilelt  p15.b, xzr, x0
// CHECK-INST: whilelt	p15.b, xzr, x0
// CHECK-ENCODING: [0xef,0x17,0x20,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ef 17 20 25 <unknown>

whilelt  p15.b, x0, xzr
// CHECK-INST: whilelt	p15.b, x0, xzr
// CHECK-ENCODING: [0x0f,0x14,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 14 3f 25 <unknown>

whilelt  p15.b, wzr, w0
// CHECK-INST: whilelt	p15.b, wzr, w0
// CHECK-ENCODING: [0xef,0x07,0x20,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ef 07 20 25 <unknown>

whilelt  p15.b, w0, wzr
// CHECK-INST: whilelt	p15.b, w0, wzr
// CHECK-ENCODING: [0x0f,0x04,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 04 3f 25 <unknown>

whilelt  p15.h, x0, xzr
// CHECK-INST: whilelt	p15.h, x0, xzr
// CHECK-ENCODING: [0x0f,0x14,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 14 7f 25 <unknown>

whilelt  p15.h, w0, wzr
// CHECK-INST: whilelt	p15.h, w0, wzr
// CHECK-ENCODING: [0x0f,0x04,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 04 7f 25 <unknown>

whilelt  p15.s, x0, xzr
// CHECK-INST: whilelt	p15.s, x0, xzr
// CHECK-ENCODING: [0x0f,0x14,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 14 bf 25 <unknown>

whilelt  p15.s, w0, wzr
// CHECK-INST: whilelt	p15.s, w0, wzr
// CHECK-ENCODING: [0x0f,0x04,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 04 bf 25 <unknown>

whilelt  p15.d, w0, wzr
// CHECK-INST: whilelt	p15.d, w0, wzr
// CHECK-ENCODING: [0x0f,0x04,0xff,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 04 ff 25 <unknown>

whilelt  p15.d, x0, xzr
// CHECK-INST: whilelt	p15.d, x0, xzr
// CHECK-ENCODING: [0x0f,0x14,0xff,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0f 14 ff 25 <unknown>
