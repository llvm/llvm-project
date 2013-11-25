; RUN: llc < %s -mtriple x86_64-apple-macosx10.9.0 | FileCheck %s
; PR18023

; CHECK: movabsq $4294967296, %rcx
; CHECK: movq  %rcx, (%rax)
; CHECK: movl  $1, 4(%rax)
; CHECK: movl  $0, 4(%rax)
; CHECK: movq  $1, 4(%rax)

@c = common global i32 0, align 4
@a = common global [3 x i32] zeroinitializer, align 4
@b = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

define void @func() {
  store i32 1, i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 1), align 4
  store i32 0, i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 0), align 4
  %1 = load volatile i32* @b, align 4
  store i32 1, i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 1), align 4
  store i32 0, i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 1), align 4
  %2 = load volatile i32* @b, align 4
  store i32 1, i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 1), align 4
  store i32 0, i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 2), align 4
  %3 = load volatile i32* @b, align 4
  store i32 3, i32* @c, align 4
  %4 = load i32* getelementptr inbounds ([3 x i32]* @a, i64 0, i64 1), align 4
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32 %4)
  ret void
}

declare i32 @printf(i8*, ...)
