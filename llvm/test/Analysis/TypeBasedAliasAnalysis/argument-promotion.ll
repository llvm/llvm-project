; RUN: opt < %s -passes=argpromotion,mem2reg -S | FileCheck %s

target datalayout = "E-p:64:64:64"

; CHECK: test
; CHECK-NOT: alloca
define internal i32 @test(i32* %X, i32* %Y, i32* %Q) {
  store i32 77, i32* %Q, !tbaa !2
  %A = load i32, i32* %X, !tbaa !1
  %B = load i32, i32* %Y, !tbaa !1
  %C = add i32 %A, %B
  ret i32 %C
}

; CHECK: caller
; CHECK-NOT: alloca
define internal i32 @caller(i32* %B, i32* %Q) {
  %A = alloca i32
  store i32 78, i32* %Q, !tbaa !2
  store i32 1, i32* %A, !tbaa !1
  %C = call i32 @test(i32* %A, i32* %B, i32* %Q)
  ret i32 %C
}

; CHECK: callercaller
; CHECK-NOT: alloca
define i32 @callercaller(i32* %Q) {
  %B = alloca i32
  store i32 2, i32* %B, !tbaa !1
  store i32 79, i32* %Q, !tbaa !2
  %X = call i32 @caller(i32* %B, i32* %Q)
  ret i32 %X
}

!0 = !{!"test"}
!1 = !{!3, !3, i64 0}
!2 = !{!4, !4, i64 0}
!3 = !{!"green", !0}
!4 = !{!"blue", !0}
