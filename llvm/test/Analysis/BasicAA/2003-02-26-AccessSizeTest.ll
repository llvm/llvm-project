; This testcase makes sure that size is taken to account when alias analysis 
; is performed.  It is not legal to delete the second load instruction because
; the value computed by the first load instruction is changed by the store.

; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s

define i32 @test() {
; CHECK: %Y.DONOTREMOVE = load i32, ptr %A
; CHECK: %Z = sub i32 0, %Y.DONOTREMOVE
  %A = alloca i32
  store i32 0, ptr %A
  %X = load i32, ptr %A
  %C = getelementptr i8, ptr %A, i64 1
  store i8 1, ptr %C    ; Aliases %A
  %Y.DONOTREMOVE = load i32, ptr %A
  %Z = sub i32 %X, %Y.DONOTREMOVE
  ret i32 %Z
}

