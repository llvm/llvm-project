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

ctermeq w30, wzr
// CHECK-INST: ctermeq	w30, wzr
// CHECK-ENCODING: [0xc0,0x23,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25bf23c0 <unknown>

ctermeq wzr, w30
// CHECK-INST: ctermeq	wzr, w30
// CHECK-ENCODING: [0xe0,0x23,0xbe,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25be23e0 <unknown>

ctermeq x30, xzr
// CHECK-INST: ctermeq	x30, xzr
// CHECK-ENCODING: [0xc0,0x23,0xff,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ff23c0 <unknown>

ctermeq xzr, x30
// CHECK-INST: ctermeq	xzr, x30
// CHECK-ENCODING: [0xe0,0x23,0xfe,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25fe23e0 <unknown>
