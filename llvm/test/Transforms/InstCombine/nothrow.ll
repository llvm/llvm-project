; RUN: opt < %s -instcombine -S | not grep call
; rdar://6880732
declare double @t1(i32) readonly willreturn

define void @t2() nounwind {
  call double @t1(i32 42)  ;; dead call even though callee is not nothrow.
  ret void
}
