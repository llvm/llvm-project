; RUN: opt -dce -S < %s | FileCheck %s
; RUN: opt -passes=dce -S < %s | FileCheck %s

; CHECK-LABEL: @test
define void @test() {
; CHECK-NOT: add
  %add = add i32 1, 2
; CHECK-NOT: sub
  %sub = sub i32 %add, 1
  ret void
}
