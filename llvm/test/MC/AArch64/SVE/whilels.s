// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

whilels  p15.b, xzr, x0
// CHECK-INST: whilels	p15.b, xzr, x0
// CHECK-ENCODING: [0xff,0x1f,0x20,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 20 25 <unknown>

whilels  p15.b, x0, xzr
// CHECK-INST: whilels	p15.b, x0, xzr
// CHECK-ENCODING: [0x1f,0x1c,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 1c 3f 25 <unknown>

whilels  p15.b, wzr, w0
// CHECK-INST: whilels	p15.b, wzr, w0
// CHECK-ENCODING: [0xff,0x0f,0x20,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 0f 20 25 <unknown>

whilels  p15.b, w0, wzr
// CHECK-INST: whilels	p15.b, w0, wzr
// CHECK-ENCODING: [0x1f,0x0c,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 0c 3f 25 <unknown>

whilels  p15.h, x0, xzr
// CHECK-INST: whilels	p15.h, x0, xzr
// CHECK-ENCODING: [0x1f,0x1c,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 1c 7f 25 <unknown>

whilels  p15.h, w0, wzr
// CHECK-INST: whilels	p15.h, w0, wzr
// CHECK-ENCODING: [0x1f,0x0c,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 0c 7f 25 <unknown>

whilels  p15.s, x0, xzr
// CHECK-INST: whilels	p15.s, x0, xzr
// CHECK-ENCODING: [0x1f,0x1c,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 1c bf 25 <unknown>

whilels  p15.s, w0, wzr
// CHECK-INST: whilels	p15.s, w0, wzr
// CHECK-ENCODING: [0x1f,0x0c,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 0c bf 25 <unknown>

whilels  p15.d, w0, wzr
// CHECK-INST: whilels	p15.d, w0, wzr
// CHECK-ENCODING: [0x1f,0x0c,0xff,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 0c ff 25 <unknown>

whilels  p15.d, x0, xzr
// CHECK-INST: whilels	p15.d, x0, xzr
// CHECK-ENCODING: [0x1f,0x1c,0xff,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f 1c ff 25 <unknown>
