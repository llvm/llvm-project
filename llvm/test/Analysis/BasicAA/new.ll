; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; Don't assume that operator new without attributes does not access unrelated
; memory.

declare noalias ptr @_Znwm(i64)

; CHECK-LABEL: Function: test:
; CHECK: Both ModRef:  Ptr: i8* %p	<->  %1 = call ptr @_Znwm(i64 4)
define void @test(ptr %p) {
  call ptr @_Znwm(i64 4)
  load i8, ptr %p
  ret void
}
