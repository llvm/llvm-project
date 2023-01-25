; RUN: opt < %s -loop-reduce -lsr-filter-same-scaled-reg=true -mtriple=x86_64-unknown-linux-gnu -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.ham = type { i8, i8, [5 x i32], i64, i64, i64 }

@global = external local_unnamed_addr global %struct.ham, align 8

define void @foo() local_unnamed_addr {
bb:
  %tmp = load i64, ptr getelementptr inbounds (%struct.ham, ptr @global, i64 0, i32 3), align 8
  %tmp1 = and i64 %tmp, 1792
  %tmp2 = load i64, ptr getelementptr inbounds (%struct.ham, ptr @global, i64 0, i32 4), align 8
  %tmp3 = add i64 %tmp1, %tmp2
  %tmp4 = load ptr, ptr null, align 8
  %tmp6 = sub i64 0, %tmp3
  %tmp7 = getelementptr inbounds i8, ptr %tmp4, i64 %tmp6
  %tmp8 = inttoptr i64 0 to ptr
  br label %bb9

; Without filtering non-optimal formulae with the same ScaledReg and Scale, the strategy
; to narrow LSR search space by picking winner reg will generate only one lsr.iv and
; unoptimal result.
; CHECK-LABEL: @foo(
; CHECK: bb9:
; CHECK-NEXT: = phi ptr
; CHECK-NEXT: = phi ptr

bb9:                                              ; preds = %bb12, %bb
  %tmp10 = phi ptr [ %tmp7, %bb ], [ %tmp16, %bb12 ]
  %tmp11 = phi ptr [ %tmp8, %bb ], [ %tmp17, %bb12 ]
  br i1 false, label %bb18, label %bb12

bb12:                                             ; preds = %bb9
  %tmp13 = getelementptr inbounds i8, ptr %tmp10, i64 8
  %tmp15 = load i64, ptr %tmp13, align 1
  %tmp16 = getelementptr inbounds i8, ptr %tmp10, i64 16
  %tmp17 = getelementptr inbounds i8, ptr %tmp11, i64 16
  br label %bb9

bb18:                                             ; preds = %bb9
  %tmp19 = icmp ugt ptr %tmp11, null
  %tmp20 = getelementptr inbounds i8, ptr %tmp10, i64 8
  %tmp21 = getelementptr inbounds i8, ptr %tmp11, i64 8
  %tmp22 = select i1 %tmp19, ptr %tmp10, ptr %tmp20
  %tmp23 = select i1 %tmp19, ptr %tmp11, ptr %tmp21
  br label %bb24

bb24:                                             ; preds = %bb24, %bb18
  %tmp25 = phi ptr [ %tmp27, %bb24 ], [ %tmp22, %bb18 ]
  %tmp26 = phi ptr [ %tmp29, %bb24 ], [ %tmp23, %bb18 ]
  %tmp27 = getelementptr inbounds i8, ptr %tmp25, i64 1
  %tmp28 = load i8, ptr %tmp25, align 1
  %tmp29 = getelementptr inbounds i8, ptr %tmp26, i64 1
  store i8 %tmp28, ptr %tmp26, align 1
  %tmp30 = icmp eq ptr %tmp29, %tmp4
  br label %bb24
}
