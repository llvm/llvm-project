;; Tests that callee_type metadata attached to direct call sites are safely ignored.

; RUN: llc --call-graph-section -mtriple arm-linux-gnu < %s -stop-after=finalize-isel -o - | FileCheck --match-full-lines %s

;; Test that `calleeTypeIds` field is not present in `callSites`
; CHECK-LABEL: callSites:
; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
; CHECK-NEXT: - { bb: {{[0-9]+}}, offset: {{[0-9]+}}, fwdArgRegs: [] }
define i32 @foo(i32 %x, i32 %y) !type !0 {
entry:
  ;; Call instruction with accurate callee_type.
  ;; callee_type should be dropped seemlessly.
  %call = call i32 @fizz(i32 %x, i32 %y), !callee_type !1
  ;; Call instruction with mismatched callee_type.
  ;; callee_type should be dropped seemlessly without errors.
  %call1 = call i32 @fizz(i32 %x, i32 %y), !callee_type !3
  %add = add nsw i32 %call, %call1
  ;; Call instruction with mismatched callee_type.
  ;; callee_type should be dropped seemlessly without errors.
  %call2 = call i32 @fizz(i32 %add, i32 %y), !callee_type !3
  %sub = sub nsw i32 %add, %call2
  ret i32 %sub
}

declare !type !2 i32 @fizz(i32, i32)

!0 = !{i64 0, !"_ZTSFiiiiE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFiiiE.generalized"}
!3 = !{!4}
!4 = !{i64 0, !"_ZTSFicE.generalized"}
