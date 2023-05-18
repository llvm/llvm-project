; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i8
@b = external global i8

@c1 = constant i32 trunc (i64 sub (i64 ptrtoint (ptr @a to i64), i64 ptrtoint (ptr @c1 to i64)) to i32)
@c2 = constant [7 x i32] [i32 0, i32 0,
i32 trunc (i64 sub (i64 ptrtoint (ptr @a to i64), i64 ptrtoint (ptr getelementptr ([7 x i32], ptr @c2, i32 0, i32 2) to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr getelementptr ([7 x i32], ptr @c2, i32 0, i32 2) to i64)) to i32),
i32 trunc (i64 add (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr getelementptr ([7 x i32], ptr @c2, i32 0, i32 2) to i64)) to i32),
i32 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 1) to i32),
i32 trunc (i64 sub (i64 0, i64 ptrtoint (ptr getelementptr ([7 x i32], ptr @c2, i32 0, i32 2) to i64)) to i32)
]

; CHECK: @f1
define ptr @f1() {
  ; CHECK: ret ptr @a
  %l = call ptr @llvm.load.relative.i32(ptr @c1, i32 0)
  ret ptr %l
}

; CHECK: @f2
define ptr @f2() {
  ; CHECK: ret ptr @a
  %l = call ptr @llvm.load.relative.i32(ptr getelementptr ([7 x i32], ptr @c2, i64 0, i64 2), i32 0)
  ret ptr %l
}

; CHECK: @f3
define ptr @f3() {
  ; CHECK: ret ptr @b
  %l = call ptr @llvm.load.relative.i64(ptr getelementptr ([7 x i32], ptr @c2, i64 0, i64 2), i64 4)
  ret ptr %l
}

; CHECK: @f4
define ptr @f4() {
  ; CHECK: ret ptr %
  %l = call ptr @llvm.load.relative.i32(ptr getelementptr ([7 x i32], ptr @c2, i64 0, i64 2), i32 1)
  ret ptr %l
}

; CHECK: @f5
define ptr @f5() {
  ; CHECK: ret ptr %
  %l = call ptr @llvm.load.relative.i32(ptr zeroinitializer, i32 0)
  ret ptr %l
}

; CHECK: @f6
define ptr @f6() {
  ; CHECK: ret ptr %
  %l = call ptr @llvm.load.relative.i32(ptr getelementptr ([7 x i32], ptr @c2, i64 0, i64 2), i32 8)
  ret ptr %l
}

; CHECK: @f7
define ptr @f7() {
  ; CHECK: ret ptr %
  %l = call ptr @llvm.load.relative.i32(ptr getelementptr ([7 x i32], ptr @c2, i64 0, i64 2), i32 12)
  ret ptr %l
}

; CHECK: @f8
define ptr @f8() {
  ; CHECK: ret ptr %
  %l = call ptr @llvm.load.relative.i32(ptr getelementptr ([7 x i32], ptr @c2, i64 0, i64 2), i32 16)
  ret ptr %l
}

declare ptr @llvm.load.relative.i32(ptr, i32)
declare ptr @llvm.load.relative.i64(ptr, i64)
