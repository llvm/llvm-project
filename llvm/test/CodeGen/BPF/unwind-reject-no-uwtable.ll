; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s 2>&1 | FileCheck %s

; Only functions with uwtable take part in unwinding. A landing pad in
; a function without it would never be reached.

declare void @may_throw()
declare i32 @rust_eh_personality(i32, i64, ptr, ptr)

define void @no_uwtable() personality ptr @rust_eh_personality {
entry:
  invoke void @may_throw()
          to label %ok unwind label %lpad

ok:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

; CHECK: error: {{.*}} BPF needs uwtable on a function with a landing pad
