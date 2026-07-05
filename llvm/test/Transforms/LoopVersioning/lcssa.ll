; RUN: opt -passes=loop-versioning -S < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

define void @fill(ptr %ls1.20, ptr %ls2.21, ptr %cse3.22) {
; CHECK-LABEL: @fill(
; CHECK-NEXT:  bb1.lver.check:
; CHECK-NEXT:    [[LS1_20_PROMOTED:%.*]] = load ptr, ptr [[LS1_20:%.*]], align 8
; CHECK-NEXT:    [[LS2_21_PROMOTED:%.*]] = load ptr, ptr [[LS2_21:%.*]], align 8
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, ptr [[LS1_20_PROMOTED]], i64 -1
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = getelementptr i8, ptr [[LS1_20_PROMOTED]], i64 1
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, ptr [[LS2_21_PROMOTED]], i64 1
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr [[LS2_21_PROMOTED]], [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %bb1.ph.lver.orig, label %bb1.ph
; CHECK:       bb1.ph.lver.orig:
;
bb1.ph:
  %ls1.20.promoted = load ptr, ptr %ls1.20
  %ls2.21.promoted = load ptr, ptr %ls2.21
  br label %bb1

bb1:
  %_tmp302 = phi ptr [ %ls2.21.promoted, %bb1.ph ], [ %_tmp30, %bb1 ]
  %_tmp281 = phi ptr [ %ls1.20.promoted, %bb1.ph ], [ %_tmp28, %bb1 ]
  %_tmp14 = getelementptr i8, ptr %_tmp281, i16 -1
  %_tmp15 = load i8, ptr %_tmp14
  %add = add i8 %_tmp15, 1
  store i8 %add, ptr %_tmp281
  store i8 %add, ptr %_tmp302
  %_tmp28 = getelementptr i8, ptr %_tmp281, i16 1
  %_tmp30 = getelementptr i8, ptr %_tmp302, i16 1
  br i1 false, label %bb1, label %bb3.loopexit

bb3.loopexit:
  %_tmp30.lcssa = phi ptr [ %_tmp30, %bb1 ]
  %_tmp15.lcssa = phi i8 [ %_tmp15, %bb1 ]
  %_tmp28.lcssa = phi ptr [ %_tmp28, %bb1 ]
  store ptr %_tmp28.lcssa, ptr %ls1.20
  store i8 %_tmp15.lcssa, ptr %cse3.22
  store ptr %_tmp30.lcssa, ptr %ls2.21
  br label %bb3

bb3:
  ret void
}

define void @fill_no_null_opt(ptr %ls1.20, ptr %ls2.21, ptr %cse3.22) #0 {
; CHECK-LABEL: @fill_no_null_opt(
; CHECK-NEXT:  bb1.lver.check:
; CHECK-NEXT:    [[LS1_20_PROMOTED:%.*]] = load ptr, ptr [[LS1_20:%.*]], align 8
; CHECK-NEXT:    [[LS2_21_PROMOTED:%.*]] = load ptr, ptr [[LS2_21:%.*]], align 8
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, ptr [[LS1_20_PROMOTED]], i64 -1
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = getelementptr i8, ptr [[LS1_20_PROMOTED]], i64 1
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, ptr [[LS2_21_PROMOTED]], i64 1
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr [[LS2_21_PROMOTED]], [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %bb1.ph.lver.orig, label %bb1.ph
; CHECK:       bb1.ph.lver.orig:
;
bb1.ph:
  %ls1.20.promoted = load ptr, ptr %ls1.20
  %ls2.21.promoted = load ptr, ptr %ls2.21
  br label %bb1

bb1:
  %_tmp302 = phi ptr [ %ls2.21.promoted, %bb1.ph ], [ %_tmp30, %bb1 ]
  %_tmp281 = phi ptr [ %ls1.20.promoted, %bb1.ph ], [ %_tmp28, %bb1 ]
  %_tmp14 = getelementptr i8, ptr %_tmp281, i16 -1
  %_tmp15 = load i8, ptr %_tmp14
  %add = add i8 %_tmp15, 1
  store i8 %add, ptr %_tmp281
  store i8 %add, ptr %_tmp302
  %_tmp28 = getelementptr i8, ptr %_tmp281, i16 1
  %_tmp30 = getelementptr i8, ptr %_tmp302, i16 1
  br i1 false, label %bb1, label %bb3.loopexit

bb3.loopexit:
  %_tmp30.lcssa = phi ptr [ %_tmp30, %bb1 ]
  %_tmp15.lcssa = phi i8 [ %_tmp15, %bb1 ]
  %_tmp28.lcssa = phi ptr [ %_tmp28, %bb1 ]
  store ptr %_tmp28.lcssa, ptr %ls1.20
  store i8 %_tmp15.lcssa, ptr %cse3.22
  store ptr %_tmp30.lcssa, ptr %ls2.21
  br label %bb3

bb3:
  ret void
}

attributes #0 = { null_pointer_is_valid }
