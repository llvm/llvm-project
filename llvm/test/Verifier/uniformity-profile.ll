; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: uniformity.profile is only valid on function definitions
declare !uniformity.profile !0 void @invalid_declaration()

; CHECK: uniformity.profile must be an empty node
define void @invalid_function_payload() !uniformity.profile !1 {
  ret void
}

; CHECK: uniformity.profile must be an empty node
define void @invalid_duplicate_payload() !uniformity.profile !1 !uniformity.profile !0 {
  ret void
}

; CHECK: uniformity.profile is only valid on function definitions
define i32 @invalid_instruction(i32 %value) {
  %sum = add i32 %value, 1, !uniformity.profile !0
  ret i32 %sum
}

; CHECK: uniformity.profile is only valid on function definitions
; CHECK-NEXT: ptr @invalid_global
@invalid_global = global i32 0, !uniformity.profile !0

!0 = !{}
!1 = !{i1 false}
