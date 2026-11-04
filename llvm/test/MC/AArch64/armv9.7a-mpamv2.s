// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// Armv9.7-A FEAT_MPAMV2 Extensions
//------------------------------------------------------------------------------

msr MPAMCTL_EL1, x0
// CHECK-INST:    msr MPAMCTL_EL1, x0
// CHECK-ENCODING: [0x40,0xa5,0x18,0xd5]

msr MPAMCTL_EL12, x0
// CHECK-INST:   msr MPAMCTL_EL12, x0
// CHECK-ENCODING: [0x40,0xa5,0x1d,0xd5]

msr MPAMCTL_EL2, x0
// CHECK-INST:    msr     MPAMCTL_EL2, x0
// CHECK-ENCODING: [0x40,0xa5,0x1c,0xd5]

msr MPAMCTL_EL3, x0
// CHECK-INST:    msr     MPAMCTL_EL3, x0
// CHECK-ENCODING: [0x40,0xa5,0x1e,0xd5]

mrs x0, MPAMCTL_EL1
// CHECK-INST:        mrs     x0, MPAMCTL_EL1
// CHECK-ENCODING: [0x40,0xa5,0x38,0xd5]

mrs x0, MPAMCTL_EL12
// CHECK-INST:   mrs     x0, MPAMCTL_EL12
// CHECK-ENCODING: [0x40,0xa5,0x3d,0xd5]

mrs x0, MPAMCTL_EL2
// CHECK-INST:   mrs     x0, MPAMCTL_EL2
// CHECK-ENCODING: [0x40,0xa5,0x3c,0xd5]

mrs x0, MPAMCTL_EL3
// CHECK-INST:   mrs     x0, MPAMCTL_EL3
// CHECK-ENCODING: [0x40,0xa5,0x3e,0xd5]
