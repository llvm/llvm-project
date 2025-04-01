; Check aliasing information translation on load and store

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s --check-prefix=CHECK-EXT
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXT

; CHECK-EXT: OpCapability MemoryAccessAliasingINTEL
; CHECK-EXT: OpExtension "SPV_INTEL_memory_access_aliasing"
; CHECK-EXT: %[[#Domain1:]] = OpAliasDomainDeclINTEL
; CHECK-EXT: %[[#Scope1:]] = OpAliasScopeDeclINTEL %[[#Domain1]]
; CHECK-EXT: %[[#List1:]] = OpAliasScopeListDeclINTEL %[[#Scope1]]
; CHECK-EXT: %[[#Domain2:]] = OpAliasDomainDeclINTEL
; CHECK-EXT: %[[#Scope2:]] = OpAliasScopeDeclINTEL %[[#Domain2]]
; CHECK-EXT: %[[#List2:]] = OpAliasScopeListDeclINTEL %[[#Scope2]]
; CHECK-EXT: %[[#Domain3:]] = OpAliasDomainDeclINTEL
; CHECK-EXT: %[[#Scope2:]] = OpAliasScopeDeclINTEL %[[#Domain3]]
; CHECK-EXT: %[[#List3:]] = OpAliasScopeListDeclINTEL %[[#Scope2]]

; CHECK-EXT: %[[#]] = OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List2]]
; CHECK-EXT: %[[#]] = OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask|NoAliasINTELMask 4 %[[#List2]] %[[#List1]]
; CHECK-EXT: OpStore %[[#]] %[[#]] Aligned|NoAliasINTELMask 4 %[[#List2]]

; CHECK-EXT: %[[#]] = OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List3]]
; CHECK-EXT: %[[#]] = OpLoad %[[#]] %[[#]] Aligned|AliasScopeINTELMask 4 %[[#List3]]
; CHECK-EXT: OpStore %[[#]] %[[#]] Aligned|NoAliasINTELMask 4 %[[#List3]]

; CHECK-NO-EXT-NOT: MemoryAccessAliasingINTEL
; CHECK-NO-EXT-NOT: SPV_INTEL_memory_access_aliasing
; CHECK-NO-EXT-NOT: OpAliasDomainDeclINTEL
; CHECK-NO-EXT-NOT: OpAliasScopeDeclINTEL
; CHECK-NO-EXT-NOT: OpAliasScopeListDeclINTEL

define dso_local spir_kernel void @foo(ptr addrspace(1) noalias %_arg_, ptr addrspace(1) noalias %_arg_1, ptr addrspace(1) noalias %_arg_3) {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_ to ptr addrspace(4)
  %1 = addrspacecast ptr addrspace(1) %_arg_1 to ptr addrspace(4)
  %2 = addrspacecast ptr addrspace(1) %_arg_3 to ptr addrspace(4)
  %3 = load i32, ptr addrspace(4) %0, align 4, !alias.scope !1
  %4 = load i32, ptr addrspace(4) %1, align 4, !alias.scope !1, !noalias !7
  %add.i = add nsw i32 %4, %3
  store i32 %add.i, ptr addrspace(4) %2, align 4, !noalias !1
  ret void
}

define dso_local spir_kernel void @boo(ptr addrspace(1) noalias %_arg_, ptr addrspace(1) noalias %_arg_1, ptr addrspace(1) noalias %_arg_3, i32 %_arg_5) {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_ to ptr addrspace(4)
  %1 = addrspacecast ptr addrspace(1) %_arg_1 to ptr addrspace(4)
  %2 = addrspacecast ptr addrspace(1) %_arg_3 to ptr addrspace(4)
  %3 = load i32, ptr addrspace(4) %0, align 4, !alias.scope !4
  %4 = load i32, ptr addrspace(4) %1, align 4, !alias.scope !4
  %add.i = add i32 %3, %_arg_5
  %add3.i = add i32 %add.i, %4
  store i32 %add3.i, ptr addrspace(4) %2, align 4, !noalias !4
  ret void
}

!1 = !{!2}
!2 = distinct !{!2, !3, !"foo: %this"}
!3 = distinct !{!3, !"foo"}
!4 = !{!5}
!5 = distinct !{!5, !6, !"boo: %this"}
!6 = distinct !{!6, !"boo"}
!7 = !{!8}
!8 = distinct !{!8, !9, !"foo: %this"}
!9 = distinct !{!9, !"foo"}
