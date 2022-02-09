; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-unknown-linux-gnu"

@a = external global i8

@c1 = constant [3 x i32] [i32 0, i32 0,
i32 sub (i32 ptrtoint (i8* @a to i32), i32 ptrtoint (i32* getelementptr ([3 x i32], [3 x i32]* @c1, i32 0, i32 2) to i32))
]

; CHECK: @f1
define i8* @f1() {
  ; CHECK: ret i8* @a
  %l = call i8* @llvm.load.relative.i32(i8* bitcast (i32* getelementptr ([3 x i32], [3 x i32]* @c1, i32 0, i32 2) to i8*), i32 0)
  ret i8* %l
}

declare i8* @llvm.load.relative.i32(i8*, i32)
