; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

define void @uniform_branch(i1 %condition) {
; CHECK-LABEL: define void @uniform_branch(
; CHECK: br i1 %condition, label %then, label %else, !branch.uniformity.profile !0
  br i1 %condition, label %then, label %else, !branch.uniformity.profile !0
then:
  ret void
else:
  ret void
}

; CHECK: !0 = !{}
!0 = !{}
