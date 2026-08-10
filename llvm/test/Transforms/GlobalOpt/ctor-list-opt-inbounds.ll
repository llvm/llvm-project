; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; Don't get fooled by the inbounds keyword; it doesn't change
; the computed address.

; CHECK: @H = local_unnamed_addr global i32 2
; CHECK: @I = local_unnamed_addr global i32 2

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @CTOR, ptr null } ]
@addr = external global i32
@G = internal global [6 x [5 x i32]] zeroinitializer
@H = global i32 80
@I = global i32 90

define internal void @CTOR() {
  store i32 1, ptr @G
  store i32 2, ptr @G
  %t = load i32, ptr @G
  store i32 %t, ptr @H
  %s = load i32, ptr @G
  store i32 %s, ptr @I
  ret void
}
