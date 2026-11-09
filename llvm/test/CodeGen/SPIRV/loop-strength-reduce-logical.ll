; RUN: llc -O2 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: llc -O2 -mtriple=spirv1.6-unknown-vulkan1.3-compute -enable-new-pm %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv1.6-unknown-vulkan1.3-compute -enable-new-pm %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Array accesses should use a 32-bit or 64-bit integer loop index directly,
; including when the index is also used as a value.

; CHECK-DAG: OpName %[[#main64:]] "main"
; CHECK-DAG: OpName %[[#main32:]] "main32"
; CHECK-DAG: %[[#ulong:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#ptr:]] = OpTypePointer Private %[[#float]]
; CHECK-DAG: %[[#w:]] = OpVariable %[[#]] Private %[[#]]
; CHECK: %[[#main64]] = OpFunction
; CHECK: OpLabel
; CHECK: OpLabel
; CHECK-NEXT: OpPhi %[[#float]]
; CHECK-NEXT: %[[#i64:]] = OpPhi %[[#ulong]]
; CHECK-NEXT: OpInBoundsAccessChain %[[#ptr]] %[[#w]] %[[#i64]]{{$}}
; CHECK: OpFunctionEnd
; CHECK: %[[#main32]] = OpFunction
; CHECK: %[[#i32:]] = OpPhi %[[#uint]]
; CHECK-NEXT: OpInBoundsAccessChain %[[#ptr]] %[[#w]] %[[#i32]]{{$}}
; CHECK: OpFunctionEnd

@w = internal addrspace(10) constant [8 x float] [float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0]
@out = internal addrspace(10) global float 0.0

define void @main() #0 {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %body ]
  %done = icmp eq i64 %i, 8
  br i1 %done, label %exit, label %body

body:
  %f = uitofp i64 %i to float
  %p = getelementptr inbounds nuw [4 x i8], ptr addrspace(10) @w, i64 %i
  %v = load float, ptr addrspace(10) %p
  %m = fmul float %f, %v
  %sum.next = fadd float %sum, %m
  %i.next = add nuw nsw i64 %i, 1
  br label %header

exit:
  store float %sum, ptr addrspace(10) @out
  ret void
}

define void @main32() #0 {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %body ]
  %done = icmp eq i32 %i, 8
  br i1 %done, label %exit, label %body

body:
  %f = uitofp i32 %i to float
  %p = getelementptr inbounds nuw [4 x i8], ptr addrspace(10) @w, i32 %i
  %v = load float, ptr addrspace(10) %p
  %m = fmul float %f, %v
  %sum.next = fadd float %sum, %m
  %i.next = add nuw nsw i32 %i, 1
  br label %header

exit:
  store float %sum, ptr addrspace(10) @out
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
