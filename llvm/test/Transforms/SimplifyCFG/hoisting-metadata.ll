; RUN: opt -p simplifycfg -S %s | FileCheck %s

declare void @init(ptr)

define i64 @hoist_load_with_matching_pointers_and_tbaa(i1 %c) {
; CHECK-LABEL: define i64 @hoist_load_with_matching_pointers_and_tbaa(
; CHECK-SAME: i1 [[C:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP:%.*]] = alloca i64, align 8
; CHECK-NEXT:    call void @init(ptr [[TMP]])
; CHECK-NEXT:    [[P:%.*]] = load i64, ptr [[TMP]], align 8, !tbaa [[TBAA0:![0-9]+]]
; CHECK-NEXT:    ret i64 [[P]]
;
entry:
  %tmp = alloca i64, align 8
  call void @init(ptr %tmp)
  br i1 %c, label %then, label %else

then:
  %0 = load i64, ptr %tmp, align 8, !tbaa !0
  br label %exit

else:
  %1 = load i64, ptr %tmp, align 8, !tbaa !0
  br label %exit

exit:
  %p = phi i64 [ %0, %then ], [ %1, %else ]
  ret i64 %p
}

define i64 @hoist_load_with_matching_tbaa_different_pointers(i1 %c) {
; CHECK-LABEL: define i64 @hoist_load_with_matching_tbaa_different_pointers(
; CHECK-SAME: i1 [[C:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP:%.*]] = alloca i64, align 8
; CHECK-NEXT:    [[TMP_1:%.*]] = alloca i64, align 8
; CHECK-NEXT:    call void @init(ptr [[TMP]])
; CHECK-NEXT:    call void @init(ptr [[TMP_1]])
; CHECK-NEXT:    [[TMP0:%.*]] = load i64, ptr [[TMP]], align 8
; CHECK-NOT:       !tbaa
; CHECK-NEXT:    [[TMP1:%.*]] = load i64, ptr [[TMP_1]], align 8
; CHECK-NOT:       !tbaa
; CHECK-NEXT:    [[P:%.*]] = select i1 [[C]], i64 [[TMP0]], i64 [[TMP1]]
; CHECK-NEXT:    ret i64 [[P]]
;
entry:
  %tmp = alloca i64, align 8
  %tmp.1 = alloca i64, align 8
  call void @init(ptr %tmp)
  call void @init(ptr %tmp.1)
  br i1 %c, label %then, label %else

then:
  %0 = load i64, ptr %tmp, align 8, !tbaa !0
  br label %exit

else:
  %1 = load i64, ptr %tmp.1, align 8, !tbaa !0
  br label %exit

exit:
  %p = phi i64 [ %0, %then ], [ %1, %else ]
  ret i64 %p
}

define i64 @hoist_load_with_different_tbaa(i1 %c) {
; CHECK-LABEL: define i64 @hoist_load_with_different_tbaa(
; CHECK-SAME: i1 [[C:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP:%.*]] = alloca i64, align 8
; CHECK-NEXT:    call void @init(ptr [[TMP]])
; CHECK-NEXT:    [[P:%.*]] = load i64, ptr [[TMP]], align 8, !tbaa [[TBAA5:![0-9]+]]
; CHECK-NEXT:    ret i64 [[P]]
;
entry:
  %tmp = alloca i64, align 8
  call void @init(ptr %tmp)
  br i1 %c, label %then, label %else

then:
  %0 = load i64, ptr %tmp, align 8, !tbaa !0
  br label %exit

else:
  %1 = load i64, ptr %tmp, align 8, !tbaa !5
  br label %exit

exit:
  %p = phi i64 [ %0, %then ], [ %1, %else ]
  ret i64 %p
}

define i64 @hoist_different_ops(i1 %c, i64 %a) {
; CHECK-LABEL: define i64 @hoist_different_ops(
; CHECK-SAME: i1 [[C:%.*]], i64 [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP:%.*]] = alloca i64, align 8
; CHECK-NEXT:    call void @init(ptr [[TMP]])
; CHECK-NEXT:    [[TMP0:%.*]] = load i64, ptr [[TMP]], align 8
; CHECK-NOT:       !tbaa
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[A]], 123
; CHECK-NEXT:    [[P:%.*]] = select i1 [[C]], i64 [[TMP0]], i64 [[TMP1]]
; CHECK-NEXT:    ret i64 [[P]]
;
entry:
  %tmp = alloca i64, align 8
  call void @init(ptr %tmp)
  br i1 %c, label %then, label %else

then:
  %0 = load i64, ptr %tmp, align 8, !tbaa !0
  br label %exit

else:
  %1 = add i64 %a, 123
  br label %exit

exit:
  %p = phi i64 [ %0, %then ], [ %1, %else ]
  ret i64 %p
}

!0 = !{!1, !1, i64 0}
!1 = !{!"p2 long long", !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!3, !3, i64 0}
;.
; CHECK: [[TBAA0]] = !{[[META1:![0-9]+]], [[META1]], i64 0}
; CHECK: [[META1]] = !{!"p2 long long", [[META2:![0-9]+]], i64 0}
; CHECK: [[META2]] = !{!"any pointer", [[META3:![0-9]+]], i64 0}
; CHECK: [[META3]] = !{!"omnipotent char", [[META4:![0-9]+]], i64 0}
; CHECK: [[META4]] = !{!"Simple C++ TBAA"}
; CHECK: [[TBAA5]] = !{[[META3]], [[META3]], i64 0}
;.
