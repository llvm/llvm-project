; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#VEC:]] = OpTypeVector %[[#]] 4
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#VEC]]
; CHECK-DAG: %[[#NULL:]] = OpConstantNull %[[#VEC]]

; CHECK: OpStore %[[#]] %[[#NULL]] Aligned 256
; CHECK: OpLoad %[[#VEC]] %[[#]] Aligned 256

define spir_func void @test_store_align256(ptr addrspace(1) %p) addrspace(4) {
entry:
  store <4 x i64> zeroinitializer, ptr addrspace(1) %p, align 256
  ret void
}

define spir_func <4 x i64> @test_load_align256(ptr addrspace(1) %p) addrspace(4) {
entry:
  %v = load <4 x i64>, ptr addrspace(1) %p, align 256
  ret <4 x i64> %v
}
