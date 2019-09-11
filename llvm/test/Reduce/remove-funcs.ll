; Test that llvm-reduce can remove uninteresting functions as well as
; their InstCalls.
;
; RUN: rm -rf %t
; RUN: mkdir %t
; copy the test file to preserve executable bit
; RUN: cp %p/Inputs/remove-funcs.py %t/test.py
; get the python path from lit
; RUN: echo "#!" %python > %t/test.py
; then include the rest of the test script
; RUN: cat %p/Inputs/remove-funcs.py >> %t/test.py

; RUN: llvm-reduce --test %t/test.py %s -o %t/out.ll
; RUN: cat %t/out.ll | FileCheck -implicit-check-not=uninteresting %s
; REQUIRES: plugins

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
