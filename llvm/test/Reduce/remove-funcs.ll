; Test that llvm-reduce can remove uninteresting functions as well as
; their InstCalls.
;
; RUN: llvm-reduce --test %p/Inputs/remove-funcs.py %s
; RUN: cat reduced.ll | FileCheck %s
; REQUIRES: plugins, shell

; CHECK-NOT: uninteresting1()
define i32 @uninteresting1() {
entry:
  ret i32 0
}

; CHECK: interesting()
define i32 @interesting() {
entry:
  ; CHECK: call i32 @interesting()
  %call2 = call i32 @interesting()
  ; CHECK-NOT: call i32 @uninteresting1()
  %call = call i32 @uninteresting1()
  ret i32 5
}

; CHECK-NOT: uninteresting2()
define i32 @uninteresting2() {
entry:
  ret i32 0
}

; CHECK-NOT: uninteresting3()
declare void @uninteresting3()
