; Check that '__attribute__((always_inline)) inline' functions are inlined.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -inline -S | FileCheck %s 

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

@__profn_foo = linkonce_odr hidden constant [3 x i8] c"foo"

; CHECK-LABEL: @main
; CHECK-NOT: call
define i32 @main() {
entry:
  %call = call i32 @foo()
  ret i32 %call
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32) #0

; CHECK-NOT: define available_externally i32 @foo
define available_externally i32 @foo() #1 {
entry:
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret i32 0
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline }
