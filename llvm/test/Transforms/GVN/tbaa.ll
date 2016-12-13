; RUN: opt -tbaa -basicaa -gvn -S < %s | FileCheck %s

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
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGC:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test3(i8* %p, i8* %q) {
; CHECK: @test3(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGB:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !3
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test4(i8* %p, i8* %q) {
; CHECK: @test4(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !1
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test5(i8* %p, i8* %q) {
; CHECK: @test5(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !1
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test6(i8* %p, i8* %q) {
; CHECK: @test6(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
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



define i32 @test8(i32* %p, i32* %q) {
; CHECK-LABEL: test8
; CHECK-NEXT: store i32 15, i32* %p
; CHECK-NEXT: ret i32 0
; Since we know the location is invariant, we can forward the
; load across the potentially aliasing store.

  %a = load i32, i32* %q, !tbaa !10
  store i32 15, i32* %p
  %b = load i32, i32* %q, !tbaa !10
  %c = sub i32 %a, %b
  ret i32 %c
}
define i32 @test9(i32* %p, i32* %q) {
; CHECK-LABEL: test9
; CHECK-NEXT: call void @clobber()
; CHECK-NEXT: ret i32 0
; Since we know the location is invariant, we can forward the
; load across the potentially aliasing store (within the call).

  %a = load i32, i32* %q, !tbaa !10
  call void @clobber()
  %b = load i32, i32* %q, !tbaa !10
  %c = sub i32 %a, %b
  ret i32 %c
}


declare void @clobber()
declare i32 @foo(i8*) readonly

; CHECK: [[TAGC]] = !{[[TYPEC:!.*]], [[TYPEC]], i64 0}
; CHECK: [[TYPEC]] = !{!"C", [[TYPEA:!.*]]}
; CHECK: [[TYPEA]] = !{!"A", !{{.*}}}
; CHECK: [[TAGB]] = !{[[TYPEB:!.*]], [[TYPEB]], i64 0}
; CHECK: [[TYPEB]] = !{!"B", [[TYPEA]]}
; CHECK: [[TAGA]] = !{[[TYPEA]], [[TYPEA]], i64 0}
!0 = !{!5, !5, i64 0}
!1 = !{!6, !6, i64 0}
!2 = !{!"tbaa root"}
!3 = !{!7, !7, i64 0}
!4 = !{!11, !11, i64 0}
!5 = !{!"C", !6}
!6 = !{!"A", !2}
!7 = !{!"B", !6}
!8 = !{!"another root"}
!11 = !{!"scalar type", !8}


;; A TBAA structure who's only point is to have a constant location
!9 = !{!"yet another root"}
!10 = !{!"node", !9, i64 1}

