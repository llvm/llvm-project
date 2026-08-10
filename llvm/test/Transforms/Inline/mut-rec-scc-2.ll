; RUN: opt -S -passes='inline' < %s | FileCheck %s

; Make sure we don't mark calls within the same SCC as original function with noinline.
; CHECK-NOT: function-inline-cost-multiplier

define void @samescc1() {
  call void @samescc2()
  ret void
}

define void @samescc2() {
  call void @samescc3()
  ret void
}

define void @samescc3() {
  call void @samescc1()
  ret void
}
