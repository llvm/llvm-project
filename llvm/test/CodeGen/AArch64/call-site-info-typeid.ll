;; Tests that call site type ids can be extracted and set from type operand
;; bundles.

;; Verify the exact typeId value to ensure it is not garbage but the value
;; computed as the type id from the type operand bundle.
; RUN: llc --call-graph-section -mtriple aarch64-linux-gnu < %s -stop-before=finalize-isel -o - | FileCheck %s

define dso_local void @foo(i8 signext %a) !type !3 {
entry:
  ret void
}

; CHECK: name: main
define dso_local i32 @main() !type !4 {
entry:
  %retval = alloca i32, align 4
  %fp = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store ptr @foo, ptr %fp, align 8
  %0 = load ptr, ptr %fp, align 8
  ; CHECK: callSites:
  ; CHECK-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], typeId:
  ; CHECK-NEXT: 7854600665770582568 }
  call void %0(i8 signext 97) [ "callee_type"(metadata !"_ZTSFvcE.generalized") ]
  ret i32 0
}

!3 = !{i64 0, !"_ZTSFvcE.generalized"}
!4 = !{i64 0, !"_ZTSFiE.generalized"}
