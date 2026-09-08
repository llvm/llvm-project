; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - -filetype=obj | spirv-val %}

; Two noalias scopes sharing a single alias domain. Check order correctness.

; CHECK: OpCapability MemoryAccessAliasingINTEL
; CHECK: OpExtension "SPV_INTEL_memory_access_aliasing"
; CHECK: %[[#Domain:]] = OpAliasDomainDeclINTEL
; CHECK: %[[#Scope1:]] = OpAliasScopeDeclINTEL %[[#Domain]]
; CHECK: %[[#Scope2:]] = OpAliasScopeDeclINTEL %[[#Domain]]
; CHECK: %[[#List1:]] = OpAliasScopeListDeclINTEL %[[#Scope1]]
; CHECK: %[[#List2:]] = OpAliasScopeListDeclINTEL %[[#Scope2]]

define spir_kernel void @foo(ptr addrspace(4) %a, ptr addrspace(4) %b) {
entry:
  %v = load i32, ptr addrspace(4) %a, align 4, !alias.scope !0, !noalias !3
  store i32 %v, ptr addrspace(4) %b, align 4, !noalias !0
  ret void
}

!0 = !{!1}
!1 = distinct !{!1, !2, !"foo: %a"}
!2 = distinct !{!2, !"foo"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"foo: %b"}
