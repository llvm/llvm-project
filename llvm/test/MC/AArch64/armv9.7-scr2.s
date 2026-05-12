// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d  - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding  < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64  -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING

//------------------------------------------------------------------------------
// Armv9.7-A FEAT_SCR2_EL3
//------------------------------------------------------------------------------

mrs x0, SCR2_EL3
// CHECK-INST:  mrs x0, SCR2_EL3
// CHECK-ENCODING: [0x40,0x12,0x3e,0xd5]
// CHECK-UNKNOWN: d53e1240

msr SCR2_EL3, x0
// CHECK-INST: msr SCR2_EL3, x0
// CHECK-ENCODING: [0x40,0x12,0x1e,0xd5]
// CHECK-UNKNOWN: d51e1240
