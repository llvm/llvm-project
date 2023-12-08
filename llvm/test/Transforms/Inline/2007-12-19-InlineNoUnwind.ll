; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

declare i1 @extern()

define internal i32 @test() {
; CHECK-NOT: define .* @test()
entry:
  %n = call i1 @extern()
  br i1 %n, label %r, label %u

r:
  ret i32 0

u:
  unreachable
}

define i32 @caller() {
; CHECK-LABEL: define i32 @caller()
entry:
  %X = call i32 @test() nounwind
; CHECK-NOT: call i32 @test()
; CHECK: call i1 @extern() #0
; CHECK: br i1 %{{.*}}, label %[[R:.*]], label %[[U:.*]]

; CHECK: [[U]]:
; CHECK:   unreachable

; CHECK: [[R]]:
  ret i32 %X
; CHECK:   ret i32 0
}

; CHECK: attributes #0 = { nounwind }
