;; Tests that call site callee type ids can be extracted and set from
;; callee_type metadata.

;; Verify the exact calleeTypeId value to ensure it is not garbage but the value
;; computed as the type id from the callee_type metadata.
; RUN: llc --call-graph-section -mtriple arm-linux-gnu < %s -stop-before=finalize-isel -o - | FileCheck %s

declare !type !0 void @foo(i8 signext %a)

; CHECK: name: main
define dso_local i32 @main() !type !1 {
entry:
  %retval = alloca i32, align 4
  %fp = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store ptr @foo, ptr %fp, align 8
  %fp_val = load ptr, ptr %fp, align 8
  ; CHECK: callSites:
  ; CHECK-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
  ; CHECK-NEXT: [ 7854600665770582568 ] }
  call void %fp_val(i8 signext 97), !callee_type !2
  ret i32 0
}

!0 = !{i64 0, !"_ZTSFvcE.generalized"}
!1 = !{i64 0, !"_ZTSFiE.generalized"}
!2 = !{!0}
