; Aliasing metadata on OpCopyMemory adds an alignment literal and an
; alias-list ID after the memory access mask.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#List:]] = OpAliasScopeListDeclINTEL

; CHECK: %[[#]] = OpFunction
define void @copy_aliased() {
entry:
; CHECK: OpCopyMemory %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List]]
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 @dst, ptr addrspace(1) align 4 @src, i64 20, i1 false), !alias.scope !1
  ret void
}

%struct.T = type <{ float, float, float, float, float }>
@src = external dso_local addrspace(1) global %struct.T, align 4
@dst = external dso_local addrspace(1) global %struct.T, align 4

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!1 = !{!2}
!2 = distinct !{!2, !3, !"copy_aliased: %this"}
!3 = distinct !{!3, !"copy_aliased"}
