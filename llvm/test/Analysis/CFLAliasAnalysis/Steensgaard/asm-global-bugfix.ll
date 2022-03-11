; Test case for a bug where we would crash when we were requested to report
; whether two values that didn't belong to a function (i.e. two globals, etc)
; aliased.

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

@G = private unnamed_addr constant [1 x i8] c"\00", align 1

; CHECK: Function: test_no_crash
; CHECK: 0 no alias responses
define void @test_no_crash() #0 {
entry:
  call i8* asm "nop", "=r,r"(
       i8* getelementptr inbounds ([1 x i8], [1 x i8]* @G, i64 0, i64 0))
  ret void
}
