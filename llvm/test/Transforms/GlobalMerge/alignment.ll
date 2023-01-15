; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global <{ [5 x i8], [3 x i8], [2 x i32] }> <{ [5 x i8] c"\01\01\01\01\01", [3 x i8] zeroinitializer, [2 x i32] [i32 2, i32 2] }>, align 4

; CHECK: @a = internal alias [5 x i8], ptr @_MergedGlobals
@a = internal global [5 x i8] [i8 1, i8 1, i8 1, i8 1, i8 1], align 4

; CHECK: @b = internal alias [2 x i32], getelementptr inbounds (<{ [5 x i8], [3 x i8], [2 x i32] }>, ptr @_MergedGlobals, i32 0, i32 2)
@b = internal global [2 x i32] [i32 2, i32 2]

define void @use() {
  ; CHECK: load i32, ptr @_MergedGlobals
  %x = load i32, ptr @a
  ; CHECK: load i32, ptr getelementptr inbounds (<{ [5 x i8], [3 x i8], [2 x i32] }>, ptr @_MergedGlobals, i32 0, i32 2)
  %y = load i32, ptr @b
  ret void
}
