; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test untyped pointers with array accesses.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]]
define spir_kernel void @test_array_const_idx(ptr addrspace(1) %arr, ptr addrspace(1) %out) {
entry:
  %elem = getelementptr i32, ptr addrspace(1) %arr, i64 10
  %val = load i32, ptr addrspace(1) %elem, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]]
define spir_kernel void @test_array_var_idx(ptr addrspace(1) %arr, i64 %idx, ptr addrspace(1) %out) {
entry:
  %elem = getelementptr i32, ptr addrspace(1) %arr, i64 %idx
  %val = load i32, ptr addrspace(1) %elem, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_2d_array(ptr addrspace(1) %arr, ptr addrspace(1) %out) {
entry:
  ; Access arr[2][3] where arr is [10 x [10 x i32]]
  %row = getelementptr [10 x i32], ptr addrspace(1) %arr, i64 2
  %elem = getelementptr [10 x i32], ptr addrspace(1) %row, i64 0, i64 3
  %val = load i32, ptr addrspace(1) %elem, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_array_loop(ptr addrspace(1) %arr, i64 %n, ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %elem_ptr = getelementptr i32, ptr addrspace(1) %arr, i64 %i
  %elem = load i32, ptr addrspace(1) %elem_ptr, align 4
  %sum.next = add i32 %sum, %elem
  %i.next = add i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  store i32 %sum.next, ptr addrspace(1) %out, align 4
  ret void
}
