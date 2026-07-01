; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

define void @branch_metadata(i1 %cond) {
; CHECK-LABEL: define void @branch_metadata(
entry:
  br i1 %cond, label %uniform, label %divergent, !block.uniformity.profile !0
; CHECK: br i1 %cond, label %uniform, label %divergent, !block.uniformity.profile !0

uniform:
  ret void

divergent:
  br label %uniform, !block.uniformity.profile !1
; CHECK: br label %uniform, !block.uniformity.profile !1
}

; CHECK: !0 = !{i1 true}
; CHECK: !1 = !{i1 false}
!0 = !{i1 true}
!1 = !{i1 false}
