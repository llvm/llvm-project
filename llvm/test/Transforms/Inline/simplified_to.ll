; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK-LABEL: @test()
; CHECK: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}, simplified to i1 false
; CHECK:   %1 = icmp eq i32 4, 5

define i32 @test() {
  %1 = icmp eq i32 4, 5
  ret i32 0
}

define void @main() {
  %1 = call i32 @test()
  ret void
}
