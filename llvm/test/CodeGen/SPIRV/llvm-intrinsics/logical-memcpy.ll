; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[dst_var:[0-9]+]] "dst"
; CHECK: OpName %[[src_var:[0-9]+]] "src"

; CHECK: %[[#List:]] = OpAliasScopeListDeclINTEL

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
; CHECK: OpCopyMemory %[[dst_var]] %[[src_var]]
  call void @llvm.memcpy.p0.p0.i64(ptr addrspace(1) align 4 @dst, ptr addrspace(1) align 4 @src, i64 20, i1 false)
  ret void
; CHECK: OpReturn
; CHECK: OpFunctionEnd
}

; Aliasing metadata on an OpCopyMemory pointer adds parameterized operands (an
; alignment literal plus an alias-list ID) after the memory access mask; the
; printer must consume exactly those operands rather than the fixed one it
; previously special-cased for Aligned only.
%struct.T = type <{ float, float, float, float, float }>
@src2 = external dso_local addrspace(1) global %struct.T, align 4
@dst2 = external dso_local addrspace(1) global %struct.T, align 4

; CHECK: %[[#]] = OpFunction
define void @copy_aliased() {
entry:
; CHECK: OpCopyMemory %[[#]] %[[#]] AliasScopeINTELMask %[[#List]]
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 @dst2, ptr addrspace(1) align 4 @src2, i64 20, i1 false), !alias.scope !1
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

attributes #0 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }

!1 = !{!2}
!2 = distinct !{!2, !3, !"copy_aliased: %this"}
!3 = distinct !{!3, !"copy_aliased"}


