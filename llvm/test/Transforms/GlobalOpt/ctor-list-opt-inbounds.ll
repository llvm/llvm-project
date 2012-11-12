; RUN: opt < %s -globalopt -S | FileCheck %s

; Don't get fooled by the inbounds keyword; it doesn't change
; the computed address.

; CHECK: @H = global i32 2
; CHECK: @I = global i32 2

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [ { i32, void ()* } { i32 65535, void ()* @CTOR } ]
@addr = external global i32
@G = internal global [6 x [5 x i32]] zeroinitializer
@H = global i32 80
@I = global i32 90

define internal void @CTOR() {
  store i32 1, i32* getelementptr ([6 x [5 x i32]]* @G, i64 0, i64 0, i64 0)
  store i32 2, i32* getelementptr inbounds ([6 x [5 x i32]]* @G, i64 0, i64 0, i64 0)
  %t = load i32* getelementptr ([6 x [5 x i32]]* @G, i64 0, i64 0, i64 0)
  store i32 %t, i32* @H
  %s = load i32* getelementptr inbounds ([6 x [5 x i32]]* @G, i64 0, i64 0, i64 0)
  store i32 %s, i32* @I
  ret void
}
