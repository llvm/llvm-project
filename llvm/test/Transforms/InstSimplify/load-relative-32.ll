; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-unknown-linux-gnu"

@a = external global i8

@c1 = constant [3 x i32] [i32 0, i32 0,
i32 sub (i32 ptrtoint (ptr @a to i32), i32 ptrtoint (ptr getelementptr ([3 x i32], ptr @c1, i32 0, i32 2) to i32))
]

; CHECK: @f1
define ptr @f1() {
  ; CHECK: ret ptr @a
  %l = call ptr @llvm.load.relative.i32(ptr getelementptr ([3 x i32], ptr @c1, i32 0, i32 2), i32 0)
  ret ptr %l
}

declare ptr @llvm.load.relative.i32(ptr, i32)
