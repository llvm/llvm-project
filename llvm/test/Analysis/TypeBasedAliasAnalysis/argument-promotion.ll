; RUN: opt < %s -passes=argpromotion,mem2reg -S | FileCheck %s

target datalayout = "E-p:64:64:64"

; CHECK: test
; CHECK-NOT: alloca
define internal i32 @test(ptr %X, ptr %Y, ptr %Q) {
  store i32 77, ptr %Q, !tbaa !2
  %A = load i32, ptr %X, !tbaa !1
  %B = load i32, ptr %Y, !tbaa !1
  %C = add i32 %A, %B
  ret i32 %C
}

; CHECK: caller
; CHECK-NOT: alloca
define internal i32 @caller(ptr %B, ptr %Q) {
  %A = alloca i32
  store i32 78, ptr %Q, !tbaa !2
  store i32 1, ptr %A, !tbaa !1
  %C = call i32 @test(ptr %A, ptr %B, ptr %Q)
  ret i32 %C
}

; CHECK: callercaller
; CHECK-NOT: alloca
define i32 @callercaller(ptr %Q) {
  %B = alloca i32
  store i32 2, ptr %B, !tbaa !1
  store i32 79, ptr %Q, !tbaa !2
  %X = call i32 @caller(ptr %B, ptr %Q)
  ret i32 %X
}

!0 = !{!"test"}
!1 = !{!3, !3, i64 0}
!2 = !{!4, !4, i64 0}
!3 = !{!"green", !0}
!4 = !{!"blue", !0}
