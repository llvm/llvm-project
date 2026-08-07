; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spv-allow-unknown-intrinsics=llvm %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa -verify-machineinstrs %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; llvm.experimental.noalias.scope.decl carries a metadata argument and therefore
; it can't be lowered to a function call. Check that it is dropped.

; CHECK-NOT: OpFunctionCall
; CHECK: OpStore

define spir_kernel void @foo(ptr addrspace(4) %a) {
entry:
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  store float 0.0, ptr addrspace(4) %a, align 4
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)

!0 = !{!1}
!1 = distinct !{!1, !2, !"foo: %a"}
!2 = distinct !{!2, !"foo"}
