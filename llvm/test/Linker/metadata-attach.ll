; RUN: llvm-link %s -S -o - | FileCheck %s

; CHECK: @g1 = global i32 0, !attach !0
@g1 = global i32 0, !attach !0

; CHECK: @g2 = external global i32, !attach !0
@g2 = external global i32, !attach !0

; CHECK: define void @f1() !attach !0
define void @f1() !attach !0 {
  call void @f2()
  store i32 0, i32* @g2
  ret void
}

; CHECK: declare !attach !0 void @f2()
declare !attach !0 void @f2()

!0 = !{}
