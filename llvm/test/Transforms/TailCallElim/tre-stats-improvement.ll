; REQUIRES: asserts
; RUN: opt -passes=tailcallelim -stats -S < %s 2>&1 | FileCheck %s

; Verify that the enhanced escape analysis correctly identifies non-escaping 
; allocas even when their addresses are stored in other local stack slots.
; We expect all three functions to be optimized, incrementing the NumEliminated counter.

; CHECK: 3 tailcallelim - Number of tail calls removed

define i32 @func1(i32 %n, i32 %acc) {
entry:
  %a = alloca i32
  %ptr = alloca ptr
  ; The address of %a is stored in %ptr. 
  ; Our flow-sensitive analysis proves %a does not escape through %ptr 
  ; before the recursive call.
  store ptr %a, ptr %ptr
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %exit, label %recurse
recurse:
  %n_dec = sub i32 %n, 1
  %acc_inc = add i32 %acc, 1
  %res = call i32 @func1(i32 %n_dec, i32 %acc_inc)
  ret i32 %res
exit:
  ret i32 %acc
}

define i32 @func2(i32 %n, i32 %acc) {
entry:
  %b = alloca i32
  %ptr = alloca ptr
  ; Transitive tracking: even with a similar pattern in func2, 
  ; the tracker should maintain precision.
  store ptr %b, ptr %ptr
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %exit, label %recurse
recurse:
  %n_dec = sub i32 %n, 1
  %acc_inc = add i32 %acc, 1
  %res = call i32 @func2(i32 %n_dec, i32 %acc_inc)
  ret i32 %res
exit:
  ret i32 %acc
}

define i32 @func3(i32 %n, i32 %acc) {
entry:
  %c = alloca i32
  %ptr = alloca ptr
  ; Another instance to ensure the stats counter increments correctly 
  ; across multiple successful TRE applications.
  store ptr %c, ptr %ptr
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %exit, label %recurse
recurse:
  %n_dec = sub i32 %n, 1
  %acc_inc = add i32 %acc, 1
  %res = call i32 @func3(i32 %n_dec, i32 %acc_inc)
  ret i32 %res
exit:
  ret i32 %acc
}
