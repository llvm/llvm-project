; This testcase makes sure that size is taken to account when alias analysis 
; is performed. The stores rights to parts of the second load

; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s

define i32 @test() {
; CHECK:      @test
; CHECK-NEXT: ret i32 -256

  %A = alloca i32
  store i32 0, ptr %A
  %X = load i32, ptr %A
  %C = getelementptr i8, ptr %A, i64 1
  store i8 1, ptr %C    ; Aliases %A
  %D = load i32, ptr %A
  %Z = sub i32 %X, %D
  ret i32 %Z
}

