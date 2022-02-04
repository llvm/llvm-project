; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

define <2 x i32> @sdiv(<2 x i32> %x) {
; CHECK-LABEL: @sdiv(
  %div = sdiv <2 x i32> %x, <i32 1, i32 1>
  ret <2 x i32> %div
; CHECK: ret <2 x i32> %x
}
