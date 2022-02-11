; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i8
@b = external global i8

@c1 = constant i32 trunc (i64 sub (i64 ptrtoint (i8* @a to i64), i64 ptrtoint (i32* @c1 to i64)) to i32)
@c2 = constant [7 x i32] [i32 0, i32 0,
i32 trunc (i64 sub (i64 ptrtoint (i8* @a to i64), i64 ptrtoint (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i32 0, i32 2) to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (i8* @b to i64), i64 ptrtoint (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i32 0, i32 2) to i64)) to i32),
i32 trunc (i64 add (i64 ptrtoint (i8* @b to i64), i64 ptrtoint (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i32 0, i32 2) to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (i8* @b to i64), i64 1) to i32),
i32 trunc (i64 sub (i64 0, i64 ptrtoint (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i32 0, i32 2) to i64)) to i32)
]

; CHECK: @f1
define i8* @f1() {
  ; CHECK: ret i8* @a
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* @c1 to i8*), i32 0)
  ret i8* %l
}

; CHECK: @f2
define i8* @f2() {
  ; CHECK: ret i8* @a
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i64 0, i64 2) to i8*), i32 0)
  ret i8* %l
}

; CHECK: @f3
define i8* @f3() {
  ; CHECK: ret i8* @b
  %l = call i8* @llvm.load.relative.i64(i8* bitcast (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i64 0, i64 2) to i8*), i64 4)
  ret i8* %l
}

; CHECK: @f4
define i8* @f4() {
  ; CHECK: ret i8* %
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i64 0, i64 2) to i8*), i32 1)
  ret i8* %l
}

; CHECK: @f5
define i8* @f5() {
  ; CHECK: ret i8* %
  %l = call i8* @llvm.load.relative.i32(i8* zeroinitializer, i32 0)
  ret i8* %l
}

; CHECK: @f6
define i8* @f6() {
  ; CHECK: ret i8* %
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i64 0, i64 2) to i8*), i32 8)
  ret i8* %l
}

; CHECK: @f7
define i8* @f7() {
  ; CHECK: ret i8* %
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i64 0, i64 2) to i8*), i32 12)
  ret i8* %l
}

; CHECK: @f8
define i8* @f8() {
  ; CHECK: ret i8* %
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* getelementptr ([7 x i32], [7 x i32]* @c2, i64 0, i64 2) to i8*), i32 16)
  ret i8* %l
}

declare i8* @llvm.load.relative.i32(i8*, i32)
declare i8* @llvm.load.relative.i64(i8*, i64)
