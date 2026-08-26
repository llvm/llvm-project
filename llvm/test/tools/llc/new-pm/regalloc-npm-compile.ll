; REQUIRES: x86-registered-target
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-new-pm -O3 -regalloc-npm=basic -verify-machineinstrs -filetype=null %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-new-pm -O3 -regalloc-npm=pbqp -verify-machineinstrs -filetype=null %s

define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
