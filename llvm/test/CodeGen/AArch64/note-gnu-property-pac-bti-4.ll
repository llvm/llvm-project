; RUN: llc -mtriple=aarch64-linux %s               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM

define dso_local i32 @f() #0 {
entry:
  ret i32 0
}

define dso_local i32 @g() #1 {
entry:
  ret i32 0
}

attributes #0 = { "branch-target-enforcement"="true" "sign-return-address"="non-leaf" }

attributes #1 = { "branch-target-enforcement"="true" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"branch-target-enforcement", i32 0}
!1 = !{i32 1, !"sign-return-address", i32 0}
!2 = !{i32 1, !"sign-return-address-all", i32 0}
!3 = !{i32 1, !"sign-return-address-with-bkey", i32 0}

; Note is not emited if module has no properties
; ASM-NOT: .note.gnu.property