; RUN: opt < %s -passes='require<loops>,gvn' -S | FileCheck %s
; This test is checking that (a) this doesn't crash, and (b) we don't
; conclude the value of %tmp17 is available in bb1.bb15_crit_edge.
; rdar://9429882

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"
define i1 @rb_intern(ptr %foo) nounwind ssp {
; CHECK-LABEL: @rb_intern(

bb:
  %tmp = alloca ptr, align 8
  store ptr null, ptr %tmp, align 8
  store i8 undef, ptr null, align 536870912
  br label %bb1

bb1:
  br i1 undef, label %bb3, label %bb15

; CHECK: bb1:
; CHECK: [[TMP:%.*]] = phi ptr [ %tmp14, %bb10 ], [ null, %bb ]

; CHECK: bb1.bb15_crit_edge:
; CHECK: %tmp17.pre = load i8, ptr [[TMP]], align 1

bb3:
  call void @isalnum()
  br i1 undef, label %bb10, label %bb5

bb5:
  br i1 undef, label %bb10, label %bb6

bb6:
  %tmp7 = load ptr, ptr %tmp, align 8
  %tmp8 = load i8, ptr %tmp7, align 1
  %tmp9 = zext i8 %tmp8 to i64
  br i1 undef, label %bb15, label %bb10

bb10:
  %tmp11 = load ptr, ptr %tmp, align 8
  %tmp12 = load i8, ptr %tmp11, align 1
  %tmp13 = zext i8 %tmp12 to i64
  %tmp14 = getelementptr inbounds i8, ptr %foo, i64 undef
  store ptr %tmp14, ptr %tmp, align 8
  br label %bb1

bb15:
  %tmp16 = load ptr, ptr %tmp, align 8
  %tmp17 = load i8, ptr %tmp16, align 1
  %tmp18 = icmp eq i8 %tmp17, 0
  br label %bb19

; CHECK-LABEL: bb6:
; CHECK:         br i1 undef, label %bb15split, label %bb10

; CHECK-LABEL: bb15split:                                        ; preds = %bb6
; CHECK-NEXT:    br label %bb15

; CHECK-LABEL: bb15:
; CHECK:         %tmp17 = phi i8 [ %tmp12.pre3, %bb15split ], [ %tmp17.pre, %bb1.bb15_crit_edge ]

bb19:                                             ; preds = %bb15
  ret i1 %tmp18
}

declare void @isalnum() nounwind inlinehint ssp
