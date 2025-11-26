; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[dst_var:[0-9]+]] "dst"
; CHECK: OpName %[[src_var:[0-9]+]] "src"

; CHECK: %[[f32:[0-9]+]] = OpTypeFloat 32
; CHECK: %[[structS:[0-9]+]] = OpTypeStruct %[[f32]] %[[f32]] %[[f32]] %[[f32]] %[[f32]]
; CHECK: %[[ptr_crosswkgrp_structS:[0-9]+]] = OpTypePointer CrossWorkgroup %[[structS]]
%struct.S = type <{ float, float, float, float, float }>

; CHECK-DAG: %[[src_var]] = OpVariable %[[ptr_crosswkgrp_structS]] CrossWorkgroup 
@src = external dso_local addrspace(1) global %struct.S, align 4

; CHECK-DAG: %[[dst_var]] = OpVariable %[[ptr_crosswkgrp_structS]] CrossWorkgroup 
@dst = external dso_local addrspace(1) global %struct.S, align 4

; CHECK: %[[main_func:[0-9]+]] = OpFunction %{{[0-9]+}} None %{{[0-9]+}}
; CHECK: %[[entry:[0-9]+]] = OpLabel
; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define void @main() local_unnamed_addr #0 {
entry:
; CHECK: OpCopyMemory %[[dst_var]] %[[src_var]] Aligned 4
  call void @llvm.memcpy.p0.p0.i64(ptr addrspace(1) align 4 @dst, ptr addrspace(1) align 4 @src, i64 20, i1 false)
  ret void
; CHECK: OpReturn
; CHECK: OpFunctionEnd
}

attributes #0 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }


