;; Tests that call site callee type ids can be extracted and set from
;; callee_type metadata for indirect tail calls.

;; Verify the exact calleeTypeIds value to ensure it is not garbage but the value
;; computed as the type id from the callee_type operand bundle.
; RUN: llc --call-graph-section -mtriple riscv64 < %s -stop-after=finalize-isel -o - | FileCheck --match-full-lines %s
; RUN: llc --call-graph-section -mtriple riscv32 < %s -stop-after=finalize-isel -o - | FileCheck --match-full-lines %s

define i32 @check_tailcall(ptr %func, i8 %x) !type !0 {
entry:
  ; CHECK: callSites:
  ; CHECK-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
  ; CHECK-NEXT: [ 3498816979441845844 ] }
  %call = tail call i32 %func(i8 signext %x), !callee_type !1
  ret i32 %call
}

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
