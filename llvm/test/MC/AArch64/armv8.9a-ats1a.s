// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefixes=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



at s1e1a, x1
// CHECK-INST: at s1e1a, x1
// CHECK-ENCODING: encoding: [0x41,0x79,0x08,0xd5]
// CHECK-UNKNOWN:  d5087941 at s1e1a, x1

at s1e2a, x1
// CHECK-INST: at s1e2a, x1
// CHECK-ENCODING: encoding: [0x41,0x79,0x0c,0xd5]
// CHECK-UNKNOWN:  d50c7941 at s1e2a, x1

at s1e3a, x1
// CHECK-INST: at s1e3a, x1
// CHECK-ENCODING: encoding: [0x41,0x79,0x0e,0xd5]
// CHECK-UNKNOWN:  d50e7941 at s1e3a, x1
