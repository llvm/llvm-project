; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

define void @branch_metadata(i1 %cond) !block.uniformity.profile !0 {
; CHECK-LABEL: define void @branch_metadata(i1 %cond) !block.uniformity.profile !0 {
entry:
  br i1 %cond, label %uniform, label %divergent, !block.uniformity.profile !0, !branch.uniformity.profile !0
; CHECK: br i1 %cond, label %uniform, label %divergent, !block.uniformity.profile !0, !branch.uniformity.profile !0

uniform:
  ret void

divergent:
  br label %uniform, !block.uniformity.profile !0
; CHECK: br label %uniform, !block.uniformity.profile !0
}

; CHECK: !0 = !{}
!0 = !{}
