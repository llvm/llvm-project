; RUN: opt %s -basicaa -gvn -S -o - | FileCheck %s

define i32 @test1(i8* %p, i8* %q) {
; CHECK: @test1(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p)
; CHECK-NOT: tbaa
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p)
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test2(i8* %p, i8* %q) {
; CHECK: @test2(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa !0
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test3(i8* %p, i8* %q) {
; CHECK: @test3(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa !3
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !3
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test4(i8* %p, i8* %q) {
; CHECK: @test4(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa !1
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !1
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test5(i8* %p, i8* %q) {
; CHECK: @test5(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa !1
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !1
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test6(i8* %p, i8* %q) {
; CHECK: @test6(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa !1
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test7(i8* %p, i8* %q) {
; CHECK: @test7(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p)
; CHECK-NOT: tbaa
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !4
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

declare i32 @foo(i8*) readonly

!0 = metadata !{metadata !"C", metadata !1}
!1 = metadata !{metadata !"A", metadata !2}
!2 = metadata !{metadata !"tbaa root", null}
!3 = metadata !{metadata !"B", metadata !1}
!4 = metadata !{metadata !"another root", null}
