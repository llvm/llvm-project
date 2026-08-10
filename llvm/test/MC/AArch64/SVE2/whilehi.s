// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

whilehi  p15.b, xzr, x0
// CHECK-INST: whilehi	p15.b, xzr, x0
// CHECK-ENCODING: [0xff,0x1b,0x20,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25201bff <unknown>

whilehi  p15.b, x0, xzr
// CHECK-INST: whilehi	p15.b, x0, xzr
// CHECK-ENCODING: [0x1f,0x18,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 253f181f <unknown>

whilehi  p15.b, wzr, w0
// CHECK-INST: whilehi	p15.b, wzr, w0
// CHECK-ENCODING: [0xff,0x0b,0x20,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25200bff <unknown>

whilehi  p15.b, w0, wzr
// CHECK-INST: whilehi	p15.b, w0, wzr
// CHECK-ENCODING: [0x1f,0x08,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 253f081f <unknown>

whilehi  p15.h, x0, xzr
// CHECK-INST: whilehi	p15.h, x0, xzr
// CHECK-ENCODING: [0x1f,0x18,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 257f181f <unknown>

whilehi  p15.h, w0, wzr
// CHECK-INST: whilehi	p15.h, w0, wzr
// CHECK-ENCODING: [0x1f,0x08,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 257f081f <unknown>

whilehi  p15.s, x0, xzr
// CHECK-INST: whilehi	p15.s, x0, xzr
// CHECK-ENCODING: [0x1f,0x18,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25bf181f <unknown>

whilehi  p15.s, w0, wzr
// CHECK-INST: whilehi	p15.s, w0, wzr
// CHECK-ENCODING: [0x1f,0x08,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25bf081f <unknown>

whilehi  p15.d, w0, wzr
// CHECK-INST: whilehi	p15.d, w0, wzr
// CHECK-ENCODING: [0x1f,0x08,0xff,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25ff081f <unknown>

whilehi  p15.d, x0, xzr
// CHECK-INST: whilehi	p15.d, x0, xzr
// CHECK-ENCODING: [0x1f,0x18,0xff,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25ff181f <unknown>
