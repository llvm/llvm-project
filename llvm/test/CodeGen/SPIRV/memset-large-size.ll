; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A regression test that checks if memset with a size larger exceeding i8 range
; does not crash the backend.

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#SIZE:]] = OpConstant %[[#INT32]] 444
; CHECK-DAG: %[[#ARRTY:]] = OpTypeArray %[[#INT8]] %[[#SIZE]]
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#ARRTY]]
; CHECK-DAG: %[[#LEN:]] = OpConstant %[[#INT64]] 444
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#LEN]] Aligned 1

define spir_func void @test_memset_large(ptr addrspace(4) %p) addrspace(4) {
entry:
  call addrspace(4) void @llvm.memset.p4.i64(ptr addrspace(4) %p, i8 0, i64 444, i1 false)
  ret void
}

declare void @llvm.memset.p4.i64(ptr addrspace(4) writeonly captures(none), i8, i64, i1 immarg) addrspace(4)
