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

whilege  p15.b, xzr, x0
// CHECK-INST: whilege	p15.b, xzr, x0
// CHECK-ENCODING: [0xef,0x13,0x20,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 252013ef <unknown>

whilege  p15.b, x0, xzr
// CHECK-INST: whilege	p15.b, x0, xzr
// CHECK-ENCODING: [0x0f,0x10,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 253f100f <unknown>

whilege  p15.b, wzr, w0
// CHECK-INST: whilege	p15.b, wzr, w0
// CHECK-ENCODING: [0xef,0x03,0x20,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 252003ef <unknown>

whilege  p15.b, w0, wzr
// CHECK-INST: whilege	p15.b, w0, wzr
// CHECK-ENCODING: [0x0f,0x00,0x3f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 253f000f <unknown>

whilege  p15.h, x0, xzr
// CHECK-INST: whilege	p15.h, x0, xzr
// CHECK-ENCODING: [0x0f,0x10,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 257f100f <unknown>

whilege  p15.h, w0, wzr
// CHECK-INST: whilege	p15.h, w0, wzr
// CHECK-ENCODING: [0x0f,0x00,0x7f,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 257f000f <unknown>

whilege  p15.s, x0, xzr
// CHECK-INST: whilege	p15.s, x0, xzr
// CHECK-ENCODING: [0x0f,0x10,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25bf100f <unknown>

whilege  p15.s, w0, wzr
// CHECK-INST: whilege	p15.s, w0, wzr
// CHECK-ENCODING: [0x0f,0x00,0xbf,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25bf000f <unknown>

whilege  p15.d, w0, wzr
// CHECK-INST: whilege	p15.d, w0, wzr
// CHECK-ENCODING: [0x0f,0x00,0xff,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25ff000f <unknown>

whilege  p15.d, x0, xzr
// CHECK-INST: whilege	p15.d, x0, xzr
// CHECK-ENCODING: [0x0f,0x10,0xff,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 25ff100f <unknown>
