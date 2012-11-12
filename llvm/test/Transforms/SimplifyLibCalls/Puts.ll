; Test that the PutsOptimizer works correctly
; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

target datalayout = "-p:64:64:64"

@.str = private constant [1 x i8] zeroinitializer

declare i32 @puts(i8*)

define void @foo() {
entry:
; CHECK: call i32 @putchar(i32 10)
  %call = call i32 @puts(i8* getelementptr inbounds ([1 x i8]* @.str, i32 0, i32 0))
  ret void
}
