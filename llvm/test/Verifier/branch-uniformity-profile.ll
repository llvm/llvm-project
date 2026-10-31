; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: branch.uniformity.profile is only valid on conditional branches
declare !branch.uniformity.profile !0 void @invalid_declaration()

; CHECK: branch.uniformity.profile is only valid on conditional branches
define void @invalid_function() !branch.uniformity.profile !0 {
  ret void
}

; CHECK: branch.uniformity.profile must be an empty node
define void @invalid_payload(i1 %condition) {
  br i1 %condition, label %then, label %else, !branch.uniformity.profile !1
then:
  ret void
else:
  ret void
}

; CHECK: branch.uniformity.profile is only valid on conditional branches
define void @invalid_unconditional_branch() {
  br label %exit, !branch.uniformity.profile !0
exit:
  ret void
}

; CHECK: branch.uniformity.profile is only valid on conditional branches
define i32 @invalid_select(i1 %condition) {
  %value = select i1 %condition, i32 1, i32 2, !branch.uniformity.profile !0
  ret i32 %value
}

; CHECK: branch.uniformity.profile is only valid on conditional branches
define void @invalid_return() {
  ret void, !branch.uniformity.profile !0
}

; CHECK: branch.uniformity.profile is only valid on conditional branches
define void @invalid_switch(i32 %value) {
  switch i32 %value, label %exit [i32 0, label %zero], !branch.uniformity.profile !0
zero:
  ret void
exit:
  ret void
}

; CHECK: branch.uniformity.profile is only valid on conditional branches
; CHECK-NEXT: ptr @invalid_global
@invalid_global = global i32 0, !branch.uniformity.profile !0

!0 = !{}
!1 = !{i1 true}
