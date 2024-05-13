; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

@a = addrspace(2) constant i32 1, align 4

; CHECK-DAG: OpCapability Kernel
