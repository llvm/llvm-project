; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @select1(i32 %x, i1 %b) {
; CHECK: @select1
  %rhs = select i1 %b, i32 %x, i32 1
  %rem = srem i32 %x, %rhs
  ret i32 %rem
; CHECK: ret i32 0
}

define i32 @select2(i32 %x, i1 %b) {
; CHECK: @select2
  %rhs = select i1 %b, i32 %x, i32 1
  %rem = urem i32 %x, %rhs
  ret i32 %rem
; CHECK: ret i32 0
}
