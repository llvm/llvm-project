; RUN: opt %s -basicaa -dse -S | FileCheck %s

declare void @test1f()

define void @test1(i32* noalias %p) {
       store i32 1, i32* %p
       call void @test1f()
       store i32 2, i32 *%p
       ret void
; CHECK: define void @test1
; CHECK-NOT: store
; CHECK-NEXT: call void
; CHECK-NEXT: store i32 2
; CHECK-NEXT: ret void
}
