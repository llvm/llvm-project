; RUN: opt -S -passes=globalsplit %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @global =
@global = internal constant { [2 x ptr], [1 x ptr] } {
  [2 x ptr] [ptr @f, ptr @g],
  [1 x ptr] [ptr @h]
}

define ptr @f() {
  ret ptr getelementptr ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, inrange i32 0, i32 0)
}

define ptr @g() {
  ret ptr null
}

define ptr @h() {
  ret ptr null
}

!0 = !{i32 16}
