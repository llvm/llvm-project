;; Tests that call site callee type ids can be extracted and set from
;; callee_type metadata.

;; Verify the exact calleeTypeIds value to ensure it is not garbage but the value
;; computed as the type id from the callee_type metadata.
; RUN: llc --call-graph-section -mtriple aarch64-linux-gnu < %s -stop-after=finalize-isel -o - | FileCheck --match-full-lines %s

; CHECK: name: main
; CHECK: callSites:
; CHECK-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
; CHECK-NEXT: [ 7854600665770582568 ] }
define i32 @main() {
entry:
  %fn = load ptr, ptr null, align 8
  call void %fn(i8 0), !callee_type !0
  ret i32 0
}

!0 = !{!1}
!1 = !{i64 0, !"_ZTSFvcE.generalized"}
