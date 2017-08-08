; RUN: opt -S -memcpyopt < %s | FileCheck %s
declare void @may_throw(i32* nocapture %x)

; CHECK-LABEL: define void @test1(
define void @test1(i32* nocapture noalias dereferenceable(4) %x) {
entry:
  %t = alloca i32, align 4
  call void @may_throw(i32* nonnull %t)
  %load = load i32, i32* %t, align 4
  store i32 %load, i32* %x, align 4
; CHECK:       %[[t:.*]] = alloca i32, align 4
; CHECK-NEXT:  call void @may_throw(i32* {{.*}} %[[t]])
; CHECK-NEXT:  %[[load:.*]] = load i32, i32* %[[t]], align 4
; CHECK-NEXT:  store i32 %[[load]], i32* %x, align 4
  ret void
}

declare void @always_throws()

; CHECK-LABEL: define void @test2(
define void @test2(i32* nocapture noalias dereferenceable(4) %x) {
entry:
  %t = alloca i32, align 4
  call void @may_throw(i32* nonnull %t) nounwind
  %load = load i32, i32* %t, align 4
  call void @always_throws()
  store i32 %load, i32* %x, align 4
; CHECK:       %[[t:.*]] = alloca i32, align 4
; CHECK-NEXT:  call void @may_throw(i32* {{.*}} %[[t]])
; CHECK-NEXT:  %[[load:.*]] = load i32, i32* %[[t]], align 4
; CHECK-NEXT:  call void @always_throws()
; CHECK-NEXT:  store i32 %[[load]], i32* %x, align 4
  ret void
}
