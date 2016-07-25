; RUN: opt -S -gvn-hoist < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i1 %b, i32* %x) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 2, i32* %x, align 4, !tbaa !1
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, i32* %x, align 4, !tbaa !5
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}
; CHECK-LABEL: define void @test1(
; CHECK: store i32 2, i32* %x, align 4
; CHECK-NEXT: br i1 %b

define void @test2(i1 %b, i32* %x) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %gep1 = getelementptr inbounds i32, i32* %x, i64 1
  store i32 2, i32* %gep1, align 4, !tbaa !1
  br label %if.end

if.else:                                          ; preds = %entry
  %gep2 = getelementptr inbounds i32, i32* %x, i64 1
  store i32 2, i32* %gep2, align 4, !tbaa !5
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}
; CHECK-LABEL: define void @test2(
; CHECK: %[[gep:.*]] = getelementptr inbounds i32, i32* %x, i64 1
; CHECK: store i32 2, i32* %[[gep]], align 4
; CHECK-NEXT: br i1 %b

define void @test3(i1 %b, i32* %x) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %gep1 = getelementptr inbounds i32, i32* %x, i64 1
  store i32 2, i32* %gep1, align 4, !tbaa !1
  br label %if.end

if.else:                                          ; preds = %entry
  %gep2 = getelementptr i32, i32* %x, i64 1
  store i32 2, i32* %gep2, align 4, !tbaa !5
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}
; CHECK-LABEL: define void @test3(
; CHECK: %[[gep:.*]] = getelementptr i32, i32* %x, i64 1
; CHECK: store i32 2, i32* %[[gep]], align 4
; CHECK-NEXT: br i1 %b

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"_ZTS1e", !3, i64 0}

define i32 @test4(i1 %b, i32* %y) {
entry:
  br i1 %b, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %y, align 4, !range !7
  br label %return

if.end:                                           ; preds = %entry
  %1 = load i32, i32* %y, align 4, !range !8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %0, %if.then ], [ %1, %if.end ]
  ret i32 %retval.0
}
; CHECK-LABEL: define i32 @test4(
; CHECK: %[[load:.*]] = load i32, i32* %y, align 4, !range ![[range_md:.*]]
; CHECK: %[[phi:.*]] = phi i32 [ %[[load]], %{{.*}} ], [ %[[load]], %{{.*}} ]
; CHECK: ret i32 %[[phi]]

!7 = !{i32 0, i32 2}
!8 = !{i32 3, i32 4}
; CHECK: ![[range_md]] = !{i32 0, i32 2, i32 3, i32 4}
