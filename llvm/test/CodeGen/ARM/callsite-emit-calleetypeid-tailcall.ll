;; Tests that call site callee type ids can be extracted and set from
;; callee_type metadata for indirect tail calls.

;; Verify the exact calleeTypeId value to ensure it is not garbage but the value
;; computed as the type id from the callee_type metadata.
; RUN: llc --call-graph-section -mtriple arm-linux-gnu < %s -stop-after=finalize-isel -o - | FileCheck %s

define dso_local noundef i32 @_Z13call_indirectPFicEc(ptr noundef readonly captures(none) %func, i8 noundef signext %x) local_unnamed_addr !type !0 {
entry:
  ; CHECK: callSites:
  ; CHECK-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
  ; CHECK-NEXT: [ 3498816979441845844 ] }
  %call = tail call noundef i32 %func(i8 noundef signext %x), !callee_type !1
  ret i32 %call
}

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
