// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xacquire
// CHECK: [0xf2]
    xacquire

// CHECK: xrelease
// CHECK: [0xf3]
    xrelease
