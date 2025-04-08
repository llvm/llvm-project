; Test if alias scope metadata translates well for a load/store with structure object
; type (special case as such lowering uses spv_load/store intrinsics)

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s

; CHECK: OpCapability MemoryAccessAliasingINTEL
; CHECK: OpExtension "SPV_INTEL_memory_access_aliasing"
; CHECK: %[[#Domain:]] = OpAliasDomainDeclINTEL
; CHECK: %[[#Scope:]] = OpAliasScopeDeclINTEL %[[#Domain]]
; CHECK: %[[#List:]] = OpAliasScopeListDeclINTEL %[[#Scope]]
; CHECK: %[[#]] = OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List]]
; CHECK: OpStore %[[#]] %[[#]] Aligned|NoAliasINTELMask 4 %[[#List]]

define dso_local spir_kernel void @foo(ptr addrspace(1) noalias %_arg_, ptr addrspace(1) noalias %_arg_1) {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_ to ptr addrspace(4)
  %1 = addrspacecast ptr addrspace(1) %_arg_1 to ptr addrspace(4)
  %2 = load {i32, i32}, ptr addrspace(4) %0, align 4, !alias.scope !1
  store {i32, i32} %2, ptr addrspace(4) %1, align 4, !noalias !1
  ret void
}

!1 = !{!2}
!2 = distinct !{!2, !3, !"foo: %this"}
!3 = distinct !{!3, !"foo"}
