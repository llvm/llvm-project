; RUN: opt -passes=indvars -S %s | FileCheck --check-prefix=COMMON --check-prefix=DEFAULT %s
; RUN: opt -passes=indvars -scev-range-iter-threshold=1 -S %s | FileCheck --check-prefix=COMMON --check-prefix=LIMIT %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @test(i1 %c.0, i32 %m) {
; COMMON-LABEL: @test(
; COMMON-NEXT:  entry:
; COMMON-NEXT:    br label [[OUTER_HEADER:%.*]]
; COMMON:       outer.header:
; DEFAULT-NEXT:   [[INDVARS_IV:%.*]] = phi i32 [ [[INDVARS_IV_NEXT:%.*]], [[OUTER_LATCH:%.*]] ], [ 2, [[ENTRY:%.*]] ]
; COMMON-NEXT:    [[IV_1:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IV_1_NEXT:%.*]], [[OUTER_LATCH:%.*]] ]
; COMMON-NEXT:    [[MAX_0:%.*]] = phi i32 [ 0, [[ENTRY]] ], [ [[MAX_1:%.*]], [[OUTER_LATCH]] ]
; COMMON-NEXT:    [[TMP0:%.*]] = sext i32 [[IV_1]] to i64
; COMMON-NEXT:    br label [[INNER_1:%.*]]
; COMMON:       inner.1:
; COMMON-NEXT:    [[C_1:%.*]] = icmp slt i64 0, [[TMP0]]
; COMMON-NEXT:    br i1 [[C_1]], label [[INNER_1]], label [[INNER_2_HEADER_PREHEADER:%.*]]
; COMMON:       inner.2.header.preheader:
; COMMON-NEXT:    br label [[INNER_2_HEADER:%.*]]
; COMMON:       inner.2.header:
; COMMON-NEXT:    [[IV_3:%.*]] = phi i32 [ [[IV_3_NEXT:%.*]], [[INNER_2_LATCH:%.*]] ], [ 0, [[INNER_2_HEADER_PREHEADER]] ]
; COMMON-NEXT:    br i1 [[C_0:%.*]], label [[OUTER_LATCH]], label [[INNER_2_LATCH]]
; COMMON:       inner.2.latch:
; COMMON-NEXT:    [[IV_3_NEXT]] = add i32 [[IV_3]], 1
; DEFAULT-NEXT:   [[EXITCOND:%.*]] = icmp eq i32 [[IV_3_NEXT]], [[INDVARS_IV]]
; LIMIT-NEXT:     [[EXITCOND:%.*]] = icmp ugt i32 [[IV_3]], [[IV_1]]
; COMMON-NEXT:    br i1 [[EXITCOND]], label [[OUTER_LATCH]], label [[INNER_2_HEADER]]
; COMMON:       outer.latch:
; COMMON-NEXT:    [[MAX_1]] = phi i32 [ [[M:%.*]], [[INNER_2_LATCH]] ], [ 0, [[INNER_2_HEADER]] ]
; COMMON-NEXT:    [[IV_1_NEXT]] = add nuw i32 [[IV_1]], 1
; COMMON-NEXT:    [[C_3:%.*]] = icmp ugt i32 [[IV_1]], [[MAX_0]]
; DEFAULT-NEXT:   [[INDVARS_IV_NEXT]] = add i32 [[INDVARS_IV]], 1
; COMMON-NEXT:    br i1 [[C_3]], label [[EXIT:%.*]], label [[OUTER_HEADER]], !llvm.loop [[LOOP0:![0-9]+]]
; COMMON:       exit:
; COMMON-NEXT:    ret i32 0
;
entry:
  br label %outer.header

outer.header:
  %iv.1 = phi i32 [ 0, %entry ], [ %iv.1.next, %outer.latch ]
  %iv.2 = phi i32 [ 0, %entry ], [ %iv.2.next , %outer.latch ]
  %max.0 = phi i32 [ 0, %entry ], [ %max.1, %outer.latch ]
  %0 = sext i32 %iv.1 to i64
  br label %inner.1

inner.1:
  %c.1 = icmp slt i64 0, %0
  br i1 %c.1, label %inner.1, label %inner.2.header

inner.2.header:
  %iv.3 = phi i32 [ 0, %inner.1 ], [ %iv.3.next, %inner.2.latch ]
  br i1 %c.0, label %outer.latch, label %inner.2.latch

inner.2.latch:
  %iv.3.next = add i32 %iv.3, 1
  %c.2 = icmp ugt i32 %iv.3, %iv.2
  br i1 %c.2, label %outer.latch, label %inner.2.header

outer.latch:
  %max.1 = phi i32 [ %m, %inner.2.latch ], [ %iv.3, %inner.2.header ]
  %iv.1.next = add i32 %iv.1, 1
  %iv.2.next = add i32 %iv.2, 1
  %c.3 = icmp ugt i32 %iv.2, %max.0
  br i1 %c.3, label %exit, label %outer.header, !llvm.loop !0

exit:
  ret i32 0
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
