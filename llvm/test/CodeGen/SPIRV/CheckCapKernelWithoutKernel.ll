; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s

@a = addrspace(2) constant i32 1, align 4

; CHECK-DAG: OpCapability Kernel
