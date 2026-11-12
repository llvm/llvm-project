; RUN: llvm-link %s -S -o - | FileCheck %s

@g1 = private global i8 0
; CHECK-NOT: @g1

@g2 = linkonce_odr global i8 0
; CHECK-NOT: @g2

@a1 = private alias i8, ptr @g1
; CHECK-NOT: @a1

@a2 = linkonce_odr alias i8, ptr @g2
; CHECK-NOT: @a2

define private void @f1() {
  ret void
}
; CHECK-NOT: @f1

define linkonce_odr void @f2() {
  ret void
}
; CHECK-NOT: @f2
