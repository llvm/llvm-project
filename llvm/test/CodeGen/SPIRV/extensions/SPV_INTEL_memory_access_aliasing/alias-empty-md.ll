; Check that the backend doesn't fail on a translation of empty aliasing
; metadata

; Check aliasing information translation on load and store

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s

; CHECK-NOT: MemoryAccessAliasingINTEL
; CHECK-NOT: SPV_INTEL_memory_access_aliasing
; CHECK-NOT: OpAliasDomainDeclINTEL
; CHECK-NOT: OpAliasScopeDeclINTEL
; CHECK-NOT: OpAliasScopeListDeclINTEL

define dso_local spir_kernel void @foo(ptr addrspace(1) noalias %_arg_, ptr addrspace(1) noalias %_arg_1, ptr addrspace(1) noalias %_arg_3) local_unnamed_addr {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_ to ptr addrspace(4)
  %1 = addrspacecast ptr addrspace(1) %_arg_1 to ptr addrspace(4)
  %2 = addrspacecast ptr addrspace(1) %_arg_3 to ptr addrspace(4)
  %3 = load i32, ptr addrspace(4) %0, align 4, !alias.scope !1
  %4 = load i32, ptr addrspace(4) %1, align 4, !alias.scope !1
  %add.i = add nsw i32 %4, %3
  store i32 %add.i, ptr addrspace(4) %2, align 4, !noalias !1
  ret void
}

!1 = !{}
