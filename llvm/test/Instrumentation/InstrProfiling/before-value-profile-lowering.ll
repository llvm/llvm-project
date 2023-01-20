; Test to ensure that when the only instrprof increment instruction is
; increment by step instruction and there is value profile instruction
; in front of all increment instructions in a function,
; the profile data variable is generated before value profile lowering

; RUN: opt < %s -passes=instrprof -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.instrprof.increment.step(i8*, i64, i32, i32, i64)

declare void @llvm.instrprof.value.profile(i8*, i64, i64, i32, i32)

; CHECK: @__profd_foo = private global
@__profn_foo = private constant [3 x i8] c"foo"

define i32 @foo(i32 ()* ) {
  %2 = ptrtoint i32 ()* %0 to i64
  call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i64 %2, i32 0, i32 0)
  call void @llvm.instrprof.increment.step(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0, i64 0)
  %3 = tail call i32 %0()
  ret i32 %3
}
