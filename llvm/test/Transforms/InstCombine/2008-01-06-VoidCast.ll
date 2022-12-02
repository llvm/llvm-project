; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define void @f(i16 %y) {
  ret void
}

define i32 @g(i32 %y) {
; CHECK-LABEL: @g(
; CHECK-NEXT %x = call i32 @f(i32 %y)		; <i32> [#uses=1]
  %x = call i32 @f( i32 %y )		; <i32> [#uses=1]
  ret i32 %x
}
