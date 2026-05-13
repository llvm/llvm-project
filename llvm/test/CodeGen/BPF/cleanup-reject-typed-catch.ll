; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s 2>&1 | FileCheck %s

; BPF does not support type-specific exception catches.
; Verify that a catch clause with a non-null type info is rejected.

@_ZTIi = external constant ptr

declare void @may_throw()
declare i32 @rust_personality(i32, i64, ptr, ptr)

define void @typed_catch() personality ptr @rust_personality {
entry:
  invoke void @may_throw()
          to label %ok unwind label %lpad

ok:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
    catch ptr @_ZTIi
  ret void
}

; CHECK: error: {{.*}} BPF does not support type-specific exception catches
