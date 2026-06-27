; RUN: opt < %s -passes=slsr -S | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

;;; This test encodes a regression where SCEV based reuse of an instruction
;;; derived from `or disjoint` is unsound. In the source function there is an
;;; arithmetic expression `row*4 + column` that is materialized twice. The
;;; first materialization uses `or disjoint` which can become poison if the
;;; disjointness promise is violated. The second materialization recomputes the
;;; value with ordinary `shl` and `add` so it is not poisoned on that path.
;;;
;;; The buggy optimization tries to "reuse" the first materialization on a
;;; different path. In the pointer version it reuses a GEP that is based on
;;; the `or disjoint` value. In the integer version it reuses an `add i64`
;;; based on the same value. For some inputs the original program never uses
;;; the poisoned `or disjoint` result on that path so the behavior is defined.
;;; After the transformation the reused instruction is consumed by a load or
;;; call argument which observes poison and turns the program into UB.
;;;
;;; These tests pin the behavior of ScalarEvolution::canReuseInstruction. It
;;; must reject reuse of the `or disjoint` based instruction in favor of the
;;; recomputed expression, since the candidate IR is strictly more poison
;;; generating than the SCEV expression we want to realize.

define void @invalid_gep_reuse(ptr readonly align 16 captures(none) dereferenceable(12) %0, ptr writeonly align 256 captures(none) dereferenceable(15) %1, i32 %row, i32 %column) {
; CHECK-LABEL: define void @invalid_gep_reuse(
; CHECK-SAME: ptr readonly align 16 captures(none) dereferenceable(12) [[TMP0:%.*]], ptr writeonly align 256 captures(none) dereferenceable(15) [[TMP1:%.*]], i32 [[ROW:%.*]], i32 [[COLUMN:%.*]]) {
; CHECK-NEXT:    [[TMP3:%.*]] = icmp samesign ult i32 [[COLUMN]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = shl nuw nsw i32 [[ROW]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = or disjoint i32 [[TMP4]], [[COLUMN]]
; CHECK-NEXT:    [[TMP6:%.*]] = zext nneg i32 [[TMP5]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP6]]
; CHECK-NEXT:    br i1 [[TMP3]], label %[[BB8:.*]], label %[[BB10:.*]]
; CHECK:       [[BB8]]:
; CHECK-NEXT:    [[TMP9:%.*]] = load i8, ptr [[TMP7]], align 1
; CHECK-NEXT:    br label %[[BB10]]
; CHECK:       [[BB10]]:
; CHECK-NEXT:    [[TMP11:%.*]] = phi i8 [ [[TMP9]], %[[BB8]] ], [ 1, [[TMP2:%.*]] ]
; CHECK-NEXT:    [[DOTNOT:%.*]] = icmp eq i32 [[COLUMN]], 0
; CHECK-NEXT:    [[TMP16:%.*]] = shl nuw nsw i32 [[ROW]], 2
; CHECK-NEXT:    [[TMP13:%.*]] = add nuw nsw i32 [[TMP16]], [[COLUMN]]
; CHECK-NEXT:    [[TMP17:%.*]] = zext nneg i32 [[TMP13]] to i64
; CHECK-NEXT:    [[TMP15:%.*]] = getelementptr i8, ptr [[TMP0]], i64 [[TMP17]]
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr i8, ptr [[TMP15]], i64 -1
; CHECK-NEXT:    br i1 [[DOTNOT]], label %[[BB19:.*]], label %[[BB17:.*]]
; CHECK:       [[BB17]]:
; CHECK-NEXT:    [[TMP14:%.*]] = load i8, ptr [[TMP12]], align 1
; CHECK-NEXT:    br label %[[BB19]]
; CHECK:       [[BB19]]:
; CHECK-NEXT:    ret void
;
  %3 = icmp samesign ult i32 %column, 4
  %4 = shl nuw nsw i32 %row, 2
  %5 = or disjoint i32 %4, %column
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds i8, ptr %0, i64 %6
  br i1 %3, label %8, label %10

8:                                                ; preds = %2
  %9 = load i8, ptr %7, align 1
  br label %10

10:                                               ; preds = %8, %2
  %11 = phi i8 [ %9, %8 ], [ 1, %2 ]
  %.not = icmp eq i32 %column, 0
  %12 = shl nuw nsw i32 %row, 2
  %13 = add nuw nsw i32 %12, %column
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr i8, ptr %0, i64 %14
  %16 = getelementptr i8, ptr %15, i64 -1
  br i1 %.not, label %19, label %17

17:                                               ; preds = %10
  %18 = load i8, ptr %16, align 1
  br label %19

19:                                               ; preds = %17, %10
  ret void
}

define void @invalid_add_reuse(i64 %0, ptr writeonly align 256 captures(none) dereferenceable(15) %1, i32 %row, i32 %column) {
; CHECK-LABEL: define void @invalid_add_reuse(
; CHECK-SAME: i64 [[TMP0:%.*]], ptr writeonly align 256 captures(none) dereferenceable(15) [[TMP1:%.*]], i32 [[ROW:%.*]], i32 [[COLUMN:%.*]]) {
; CHECK-NEXT:    [[TMP3:%.*]] = icmp samesign ult i32 [[COLUMN]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = shl nuw nsw i32 [[ROW]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = or disjoint i32 [[TMP4]], [[COLUMN]]
; CHECK-NEXT:    [[TMP6:%.*]] = zext nneg i32 [[TMP5]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = add i64 [[TMP0]], [[TMP6]]
; CHECK-NEXT:    br i1 [[TMP3]], label %[[BB8:.*]], label %[[BB10:.*]]
; CHECK:       [[BB8]]:
; CHECK-NEXT:    [[TMP9:%.*]] = call i64 @foo(i64 [[TMP7]])
; CHECK-NEXT:    br label %[[BB10]]
; CHECK:       [[BB10]]:
; CHECK-NEXT:    [[TMP11:%.*]] = phi i64 [ [[TMP9]], %[[BB8]] ], [ 1, [[TMP2:%.*]] ]
; CHECK-NEXT:    [[DOTNOT:%.*]] = icmp eq i32 [[COLUMN]], 0
; CHECK-NEXT:    [[TMP12:%.*]] = shl nuw nsw i32 [[ROW]], 2
; CHECK-NEXT:    [[TMP17:%.*]] = add nuw nsw i32 [[TMP12]], [[COLUMN]]
; CHECK-NEXT:    [[TMP18:%.*]] = zext nneg i32 [[TMP17]] to i64
; CHECK-NEXT:    [[TMP15:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP13:%.*]] = add i64 [[TMP15]], [[TMP18]]
; CHECK-NEXT:    [[TMP14:%.*]] = add i64 [[TMP13]], -1
; CHECK-NEXT:    br i1 [[DOTNOT]], label %[[BB20:.*]], label %[[BB18:.*]]
; CHECK:       [[BB18]]:
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @bar(i64 [[TMP14]])
; CHECK-NEXT:    br label %[[BB20]]
; CHECK:       [[BB20]]:
; CHECK-NEXT:    ret void
;
  %3 = icmp samesign ult i32 %column, 4
  %4 = shl nuw nsw i32 %row, 2
  %5 = or disjoint i32 %4, %column
  %6 = zext nneg i32 %5 to i64
  %7 = add i64 %0, %6
  br i1 %3, label %8, label %10

8:                                                ; preds = %2
  %9 = call i64 @foo(i64 %7)
  br label %10

10:                                               ; preds = %8, %2
  %11 = phi i64 [ %9, %8 ], [ 1, %2 ]
  %.not = icmp eq i32 %column, 0
  %12 = shl nuw nsw i32 %row, 2
  %13 = add nuw nsw i32 %12, %column
  %14 = zext nneg i32 %13 to i64
  %15 = mul i64 %0, 4
  %16 = add i64 %15, %14
  %17 = add i64 %16, -1
  br i1 %.not, label %20, label %18

18:                                               ; preds = %10
  %19 = call i64 @bar(i64 %17)
  br label %20

20:                                               ; preds = %18, %10
  ret void
}

declare i64 @foo(i64)
declare i64 @bar(i64)
