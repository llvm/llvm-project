; Test the case where the LoopCounter's stride equals -1.
; RUN: opt -S -passes=indvars  < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @check_step_minus_one(ptr nocapture readonly %0)  {
; CHECK-LABEL: define void @check_step_minus_one(ptr readonly captures(none) %0) {
; CHECK:       entry:
; CHECK-NEXT:  br label [[loop:.*]]
; CHECK:       loop:
; CHECK-NEXT:  [[IV:%.*]] = phi i64 [ 31, [[entry:%.*]] ], [ [[PostDec:%.*]], [[loop:%.*]] ]
; CHECK-NEXT:  [[GEP:%.*]] = getelementptr inbounds i32, ptr %0, i64 [[IV]]
; CHECK-NEXT:  [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:  [[ADD:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:  store i32 [[ADD]], ptr [[GEP]], align 4
; CHECK-NEXT:  [[PostDec:%.*]] = add nsw i64 [[IV]], -1
; CHECK-NEXT:  [[CMP:%.*]] = icmp ne i64 [[PostDec]], 6
; CHECK-NEXT:  br i1 [[CMP]], label [[loop:.*]], label [[end:.*]]
; CHECK:       end:
; CHECK-NEXT:    ret void
;
entry:                  
  br label %loop

loop:                                           
  %1 = phi i64 [ 31, %entry ], [ %6, %loop ]
  %3 = getelementptr inbounds i32, ptr %0, i64 %1
  %4 = load i32, ptr %3, align 4
  %5 = add nsw i32 %4, 1
  store i32 %5, ptr %3, align 4
  %6 = add nsw i64 %1, -1
  %7 = mul nsw i64 %6, %6
  %8 = icmp samesign ugt i64 %7, 48
  br i1 %8, label %loop, label %end

end:                                      
  ret void
}

