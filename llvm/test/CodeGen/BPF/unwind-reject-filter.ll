; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s 2>&1 | FileCheck %s

; A filter that lets some types through needs type information at run time.
; The empty one is a catch-all and is accepted, see unwind.ll.

@_ZTIi = external constant ptr

declare void @may_throw()
declare i32 @rust_eh_personality(i32, i64, ptr, ptr)

define void @typed_filter() uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @may_throw()
          to label %ok unwind label %lpad

ok:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } filter [1 x ptr] [ptr @_ZTIi]
  ret void
}

; CHECK: error: {{.*}} BPF does not support exception filters yet
