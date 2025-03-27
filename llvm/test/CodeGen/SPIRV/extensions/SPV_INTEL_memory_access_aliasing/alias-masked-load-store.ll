; Check aliasing information translation on function calls

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
; CHECK-EXT: OpDecorate %[[#Fun1:]] AliasScopeINTEL %[[#List1]]
; CHECK-EXT: OpDecorate %[[#Fun2:]] AliasScopeINTEL %[[#List1]]
; CHECK-EXT: OpDecorate %[[#Fun2]] NoAliasINTEL %[[#List2]]
; CHECK-EXT: OpDecorate %[[#Fun3:]] NoAliasINTEL %[[#List1]]
; CHECK-EXT: OpDecorate %[[#Fun4:]] AliasScopeINTEL %[[#List3]]
; CHECK-EXT: OpDecorate %[[#Fun5:]] AliasScopeINTEL %[[#List3]]
; CHECK-EXT: OpDecorate %[[#Fun6:]] NoAliasINTEL %[[#List3]]
; CHECK-EXT: %[[#Fun1]] = OpFunctionCall
; CHECK-EXT: %[[#Fun2]] = OpFunctionCall
; CHECK-EXT: %[[#Fun3]] = OpFunctionCall
; CHECK-EXT: %[[#Fun4]] = OpFunctionCall
; CHECK-EXT: %[[#Fun5]] = OpFunctionCall
; CHECK-EXT: %[[#Fun6]] = OpFunctionCall

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
  %3 = call i32 @wrappedload(ptr addrspace(4) %0), !alias.scope !1
  %4 = call i32 @wrappedload(ptr addrspace(4) %1), !alias.scope !1, !noalias !7
  %add.i = add nsw i32 %4, %3
  call void @wrappedstore(i32 %add.i, ptr addrspace(4) %2), !noalias !1
  ret void
}

define dso_local spir_kernel void @boo(ptr addrspace(1) noalias %_arg_, ptr addrspace(1) noalias %_arg_1, ptr addrspace(1) noalias %_arg_3, i32 %_arg_5) {
entry:
  %0 = addrspacecast ptr addrspace(1) %_arg_ to ptr addrspace(4)
  %1 = addrspacecast ptr addrspace(1) %_arg_1 to ptr addrspace(4)
  %2 = addrspacecast ptr addrspace(1) %_arg_3 to ptr addrspace(4)
  %3 = call i32 @wrappedload(ptr addrspace(4) %0), !alias.scope !4
  %4 = call i32 @wrappedload(ptr addrspace(4) %1), !alias.scope !4
  %add.i = add i32 %3, %_arg_5
  %add3.i = add i32 %add.i, %4
  call void @wrappedstore(i32 %add3.i, ptr addrspace(4) %2), !noalias !4
  ret void
}

define dso_local spir_func i32 @wrappedload(ptr addrspace(4) %0) {
entry:
  %1 = load i32, ptr addrspace(4) %0, align 4
  ret i32 %1
}

; Function Attrs: norecurse nounwind readnone
define dso_local spir_func void @wrappedstore(i32 %0, ptr addrspace(4) %1) {
entry:
  store i32 %0, ptr addrspace(4) %1, align 4
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
