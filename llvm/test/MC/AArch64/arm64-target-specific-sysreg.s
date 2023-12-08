// RUN: not llvm-mc -triple arm64 -mcpu=generic -show-encoding < %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-GENERIC
//
// RUN: llvm-mc -triple arm64 -mcpu=cyclone -show-encoding < %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-CYCLONE

msr CPM_IOACC_CTL_EL3, x0

// CHECK-GENERIC: error: expected writable system register or pstate
// CHECK-CYCLONE: msr CPM_IOACC_CTL_EL3, x0   // encoding: [0x00,0xf2,0x1f,0xd5]
