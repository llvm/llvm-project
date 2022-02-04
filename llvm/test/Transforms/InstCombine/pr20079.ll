; RUN: opt -S -passes=instcombine < %s | FileCheck %s
@b = internal global [1 x i32] zeroinitializer, align 4
@c = internal global i32 0, align 4

; CHECK-LABEL: @fn1
; CHECK-NEXT: ret i32 0
define i32 @fn1(i32 %a) {
  ret i32 0
}
