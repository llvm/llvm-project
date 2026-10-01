; Check that an alias scope list referenced from two functions is declared in
; each of them. The aliasing instructions are cached per MDNode, but the virtual
; registers they define belong to one MachineFunction, so a cache shared across
; functions makes the second function reference a register that does not exist
; in it.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability MemoryAccessAliasingINTEL
; CHECK: OpExtension "SPV_INTEL_memory_access_aliasing"

; Both functions use !1, so each declares its own domain, scope and list.
; CHECK: %[[#Domain1:]] = OpAliasDomainDeclINTEL
; CHECK: %[[#Domain2:]] = OpAliasDomainDeclINTEL
; CHECK: %[[#Scope1:]] = OpAliasScopeDeclINTEL %[[#Domain1]]
; CHECK: %[[#Scope2:]] = OpAliasScopeDeclINTEL %[[#Domain2]]
; CHECK: %[[#List1:]] = OpAliasScopeListDeclINTEL %[[#Scope1]]
; CHECK: %[[#List2:]] = OpAliasScopeListDeclINTEL %[[#Scope2]]

; One load per function, each using a different one of the two lists. Which
; function gets which id is not fixed, so match them in either order.
; CHECK-DAG: OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List1]]
; CHECK-DAG: OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List2]]

define spir_kernel void @foo(ptr addrspace(1) noalias %in, ptr addrspace(1) noalias %out) {
entry:
  %src = addrspacecast ptr addrspace(1) %in to ptr addrspace(4)
  %dst = addrspacecast ptr addrspace(1) %out to ptr addrspace(4)
  %val = load i32, ptr addrspace(4) %src, align 4, !alias.scope !1
  store i32 %val, ptr addrspace(4) %dst, align 4
  ret void
}

define spir_kernel void @bar(ptr addrspace(1) noalias %in, ptr addrspace(1) noalias %out) {
entry:
  %src = addrspacecast ptr addrspace(1) %in to ptr addrspace(4)
  %dst = addrspacecast ptr addrspace(1) %out to ptr addrspace(4)
  %val = load i32, ptr addrspace(4) %src, align 4, !alias.scope !1
  store i32 %val, ptr addrspace(4) %dst, align 4
  ret void
}

!1 = !{!2}
!2 = distinct !{!2, !3, !"shared: %in"}
!3 = distinct !{!3, !"shared"}
