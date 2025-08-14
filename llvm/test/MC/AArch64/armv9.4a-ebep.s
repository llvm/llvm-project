// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


mrs x2, PM
// CHECK-INST: mrs x2, PM
// CHECK-ENCODING: encoding: [0x22,0x43,0x38,0xd5]
// CHECK-UNKNOWN:  d5384322 mrs x2, PM

mrs x3, PM
// CHECK-INST: mrs x3, PM
// CHECK-ENCODING: encoding: [0x23,0x43,0x38,0xd5]
// CHECK-UNKNOWN:  d5384323 mrs x3, PM

msr PM, x3
// CHECK-INST: msr PM, x3
// CHECK-ENCODING: encoding: [0x23,0x43,0x18,0xd5]
// CHECK-UNKNOWN:  d5184323 msr PM, x3

msr PM, x6
// CHECK-INST: msr PM, x6
// CHECK-ENCODING: encoding: [0x26,0x43,0x18,0xd5]
// CHECK-UNKNOWN:  d5184326 msr PM, x6

msr PM, #0
// CHECK-INST: msr PM, #0
// CHECK-ENCODING: encoding: [0x1f,0x42,0x01,0xd5]
// CHECK-UNKNOWN:  d501421f msr PM, #0

msr PM, #1
// CHECK-INST: msr PM, #1
// CHECK-ENCODING: encoding: [0x1f,0x43,0x01,0xd5]
// CHECK-UNKNOWN:  d501431f msr PM, #1

