; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; ModuleID = 'test.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-freebsd"

@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1
@.str1 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

define void @foo() nounwind {
entry:
; CHECK: crxor 6, 6, 6
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 1)
; CHECK: creqv 6, 6, 6
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str1, i32 0, i32 0), double 1.100000e+00)
  ret void
}

declare i32 @printf(i8*, ...)
