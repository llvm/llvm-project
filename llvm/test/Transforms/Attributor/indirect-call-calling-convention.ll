; RUN: opt -passes=attributor -S %s | FileCheck %s
; RUN: opt -passes=attributor,instcombine -S %s | FileCheck %s --check-prefix=IC

; The callbacks' return types differ from the indirect calls in dispatch, so
; the calls cannot be promoted and are replaced with new direct calls instead.
; The new calls must keep the calling convention of the indirect calls, both
; when there is one possible callee and when there are multiple callees.
; Otherwise InstCombine sees a calling-convention mismatch with the callee and
; replaces the call with a store to poison.

@g = global i32 0

define spir_func i32 @callback1() {
  store i32 1, ptr @g
  ret i32 1
}

define spir_func i32 @callback2() {
  store i32 2, ptr @g
  ret i32 2
}

define internal void @dispatch(ptr %fn1, ptr %fn2) {
; CHECK-LABEL: define internal void @dispatch(
; CHECK:         call spir_func void @callback1()
; CHECK-DAG:     call spir_func void @callback1()
; CHECK-DAG:     call spir_func void @callback2()
;
; IC-LABEL:    define internal void @dispatch(
; IC-NOT:        store i1 true, ptr poison
; IC:            call spir_func i32 @callback1()
; IC-NOT:        store i1 true, ptr poison
; IC:            call spir_func i32 @callback2()
; IC-NOT:        store i1 true, ptr poison
; IC:            ret void
entry:
  call spir_func void %fn1()
  call spir_func void %fn2()
  ret void
}

define void @foo() {
entry:
  call void @dispatch(ptr @callback1, ptr @callback1)
  call void @dispatch(ptr @callback1, ptr @callback2)
  ret void
}
