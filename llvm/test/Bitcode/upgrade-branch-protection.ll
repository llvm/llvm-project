;; Test that module flags "branch-target-enforcement" and "sign-return-address"  can be upgraded to
;; are upgraded from Error to Min.

; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"branch-target-enforcement", i32 1}
!1 = !{i32 1, !"sign-return-address", i32 1}
!2 = !{i32 1, !"sign-return-address-all", i32 1}
!3 = !{i32 1, !"sign-return-address-with-bkey", i32 1}

;CHECK: !0 = !{i32 8, !"branch-target-enforcement", i32 1}
;CHECK: !1 = !{i32 8, !"sign-return-address", i32 1}
;CHECK: !2 = !{i32 8, !"sign-return-address-all", i32 1}
;CHECK: !3 = !{i32 8, !"sign-return-address-with-bkey", i32 1}