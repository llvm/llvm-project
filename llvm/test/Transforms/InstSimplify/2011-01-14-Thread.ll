; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @shift_select(i1 %cond) {
; CHECK-LABEL: @shift_select(
  %s = select i1 %cond, i32 0, i32 1
  %r = lshr i32 %s, 1
  ret i32 %r
; CHECK: ret i32 0
}
