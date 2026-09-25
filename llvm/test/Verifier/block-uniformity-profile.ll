; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: block.uniformity.profile is only valid on terminators
declare !block.uniformity.profile !0 void @invalid_declaration()

; CHECK: block.uniformity.profile is only valid on terminators
define void @invalid_function() !block.uniformity.profile !0 {
  ret void
}

; CHECK: block.uniformity.profile must be an empty node
define void @invalid_terminator_payload() {
  ret void, !block.uniformity.profile !1
}

; CHECK: block.uniformity.profile is only valid on terminators
define i32 @invalid_instruction(i32 %value) {
  %sum = add i32 %value, 1, !block.uniformity.profile !0
  ret i32 %sum
}

; CHECK: block.uniformity.profile is only valid on terminators
; CHECK-NEXT: ptr @invalid_global
@invalid_global = global i32 0, !block.uniformity.profile !0

!0 = !{}
!1 = !{i1 false}
