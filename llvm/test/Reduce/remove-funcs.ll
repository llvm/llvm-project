; Test that llvm-reduce can remove uninteresting functions as well as
; their InstCalls.
;
; RUN: rm -rf %t
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-funcs.py %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting %s

define i32 @uninteresting1() {
entry:
  ret i32 0
}

; CHECK: interesting()
define i32 @interesting() {
entry:
  ; CHECK: call i32 @interesting()
  %call2 = call i32 @interesting()
  %call = call i32 @uninteresting1()
  ret i32 5
}

define i32 @uninteresting2() {
entry:
  ret i32 0
}

declare void @uninteresting3()
