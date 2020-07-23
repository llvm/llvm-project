;; Test strip branch_weight if operand number doesn't match.
;; Input bitcode is generated from:
;; define void @test(i1 %0) {
;;   br i1 %0, label %2, label %3, !prof !0
;; 2:
;;   br i1 %0, label %4, label %3, !prof !1
;; 3:
;;   unreachable
;; 4:
;;   ret void
;; }
;;!0 = !{!"branch_weights", i32 1, i32 2}
;;!1 = !{!"branch_weights", i32 1, i32 2, i32 3}

; RUN: llvm-dis %S/Inputs/branch-weight.bc -o - | FileCheck %s
; CHECK: !prof !0
; CHECK: !0 = !{!"branch_weights", i32 1, i32 2}
; CHECK-NOT: !prof !1
; CHECK-NOT: !1 = !{!"branch_weights", i32 1, i32 2, i32 3}
