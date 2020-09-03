; RUN: opt -tbaa -basic-aa -gvn -S < %s | FileCheck %s

define i32 @test1(i8* %p, i8* %q) {
; CHECK-LABEL: @test1(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p)
; CHECK-NOT: tbaa
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p)
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test2(i8* %p, i8* %q) {
; CHECK-LABEL: @test2(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGC:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test3(i8* %p, i8* %q) {
; CHECK-LABEL: @test3(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGB:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !3
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test4(i8* %p, i8* %q) {
; CHECK-LABEL: @test4(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !1
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test5(i8* %p, i8* %q) {
; CHECK-LABEL: @test5(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !1
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test6(i8* %p, i8* %q) {
; CHECK-LABEL: @test6(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test7(i8* %p, i8* %q) {
; CHECK-LABEL: @test7(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p)
; CHECK-NOT: tbaa
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !4
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test8(i32* %p, i32* %q) {
; CHECK-LABEL: @test8
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
; CHECK-LABEL: @test9
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

define i32 @test10(i8* %p, i8* %q) {
; If one access encloses the other, then the merged access is the enclosed one
; and not just the common final access type.
; CHECK-LABEL: @test10
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAG_X_i:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !15  ; TAG_X_i
  %b = call i32 @foo(i8* %p), !tbaa !19  ; TAG_Y_x_i
  %c = add i32 %a, %b
  ret i32 %c
}

declare void @clobber()
declare i32 @foo(i8*) readonly

; CHECK-DAG: [[TAGC]] = !{[[TYPEC:!.*]], [[TYPEC]], i64 0}
; CHECK-DAG: [[TYPEC]] = !{!"C", [[TYPEA:!.*]]}
; CHECK-DAG: [[TYPEA]] = !{!"A", !{{.*}}}
; CHECK-DAG: [[TAGB]] = !{[[TYPEB:!.*]], [[TYPEB]], i64 0}
; CHECK-DAG: [[TYPEB]] = !{!"B", [[TYPEA]]}
; CHECK-DAG: [[TAGA]] = !{[[TYPEA]], [[TYPEA]], i64 0}
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

; CHECK-DAG: [[TAG_X_i]] = !{[[TYPE_X:!.*]], [[TYPE_int:!.*]], i64 0}
; CHECK-DAG: [[TYPE_X:!.*]] = !{!"struct X", [[TYPE_int]], i64 0}
; CHECK-DAG: [[TYPE_int]] = !{!"int", {{!.*}}, i64 0}
!15 = !{!16, !17, i64 0}            ; TAG_X_i
!16 = !{!"struct X", !17, i64 0}    ; struct X { int i; };
!17 = !{!"int", !18, i64 0}
!18 = !{!"char", !2, i64 0}

!19 = !{!20, !17, i64 0}            ; TAG_Y_x_i
!20 = !{!"struct Y", !16, i64 0}    ; struct Y { struct X x; };

; A TBAA structure who's only point is to have a constant location.
!9 = !{!"yet another root"}
!10 = !{!"node", !9, i64 1}
