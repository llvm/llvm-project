; RUN: opt -passes=loop-versioning -S < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

define void @fill(i8** %ls1.20, i8** %ls2.21, i8* %cse3.22) {
; CHECK-LABEL: @fill(
; CHECK-NEXT:  bb1.lver.check:
; CHECK-NEXT:    [[LS1_20_PROMOTED:%.*]] = load i8*, i8** [[LS1_20:%.*]], align 8
; CHECK-NEXT:    [[LS2_21_PROMOTED:%.*]] = load i8*, i8** [[LS2_21:%.*]], align 8
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, i8* [[LS1_20_PROMOTED]], i64 -1
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = getelementptr i8, i8* [[LS1_20_PROMOTED]], i64 1
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, i8* [[LS2_21_PROMOTED]], i64 1
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult i8* [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult i8* [[LS2_21_PROMOTED]], [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %bb1.ph.lver.orig, label %bb1.ph
; CHECK:       bb1.ph.lver.orig:
;
bb1.ph:
  %ls1.20.promoted = load i8*, i8** %ls1.20
  %ls2.21.promoted = load i8*, i8** %ls2.21
  br label %bb1

bb1:
  %_tmp302 = phi i8* [ %ls2.21.promoted, %bb1.ph ], [ %_tmp30, %bb1 ]
  %_tmp281 = phi i8* [ %ls1.20.promoted, %bb1.ph ], [ %_tmp28, %bb1 ]
  %_tmp14 = getelementptr i8, i8* %_tmp281, i16 -1
  %_tmp15 = load i8, i8* %_tmp14
  %add = add i8 %_tmp15, 1
  store i8 %add, i8* %_tmp281
  store i8 %add, i8* %_tmp302
  %_tmp28 = getelementptr i8, i8* %_tmp281, i16 1
  %_tmp30 = getelementptr i8, i8* %_tmp302, i16 1
  br i1 false, label %bb1, label %bb3.loopexit

bb3.loopexit:
  %_tmp30.lcssa = phi i8* [ %_tmp30, %bb1 ]
  %_tmp15.lcssa = phi i8 [ %_tmp15, %bb1 ]
  %_tmp28.lcssa = phi i8* [ %_tmp28, %bb1 ]
  store i8* %_tmp28.lcssa, i8** %ls1.20
  store i8 %_tmp15.lcssa, i8* %cse3.22
  store i8* %_tmp30.lcssa, i8** %ls2.21
  br label %bb3

bb3:
  ret void
}

define void @fill_no_null_opt(i8** %ls1.20, i8** %ls2.21, i8* %cse3.22) #0 {
; CHECK-LABEL: @fill_no_null_opt(
; CHECK-NEXT:  bb1.lver.check:
; CHECK-NEXT:    [[LS1_20_PROMOTED:%.*]] = load i8*, i8** [[LS1_20:%.*]], align 8
; CHECK-NEXT:    [[LS2_21_PROMOTED:%.*]] = load i8*, i8** [[LS2_21:%.*]], align 8
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, i8* [[LS1_20_PROMOTED]], i64 -1
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = getelementptr i8, i8* [[LS1_20_PROMOTED]], i64 1
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, i8* [[LS2_21_PROMOTED]], i64 1
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult i8* [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult i8* [[LS2_21_PROMOTED]], [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    [[SCEVGEP3:%.*]] = getelementptr i8, i8* [[LS1_20_PROMOTED]], i64 -1
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %bb1.ph.lver.orig, label %bb1.ph
; CHECK:       bb1.ph.lver.orig:
;
bb1.ph:
  %ls1.20.promoted = load i8*, i8** %ls1.20
  %ls2.21.promoted = load i8*, i8** %ls2.21
  br label %bb1

bb1:
  %_tmp302 = phi i8* [ %ls2.21.promoted, %bb1.ph ], [ %_tmp30, %bb1 ]
  %_tmp281 = phi i8* [ %ls1.20.promoted, %bb1.ph ], [ %_tmp28, %bb1 ]
  %_tmp14 = getelementptr i8, i8* %_tmp281, i16 -1
  %_tmp15 = load i8, i8* %_tmp14
  %add = add i8 %_tmp15, 1
  store i8 %add, i8* %_tmp281
  store i8 %add, i8* %_tmp302
  %_tmp28 = getelementptr i8, i8* %_tmp281, i16 1
  %_tmp30 = getelementptr i8, i8* %_tmp302, i16 1
  br i1 false, label %bb1, label %bb3.loopexit

bb3.loopexit:
  %_tmp30.lcssa = phi i8* [ %_tmp30, %bb1 ]
  %_tmp15.lcssa = phi i8 [ %_tmp15, %bb1 ]
  %_tmp28.lcssa = phi i8* [ %_tmp28, %bb1 ]
  store i8* %_tmp28.lcssa, i8** %ls1.20
  store i8 %_tmp15.lcssa, i8* %cse3.22
  store i8* %_tmp30.lcssa, i8** %ls2.21
  br label %bb3

bb3:
  ret void
}

attributes #0 = { null_pointer_is_valid }
