; Legally add nsw/nuw flag for the Trunc instruction.
; RUN: opt < %s -passes='indvars' -S  | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define dso_local void @func(ptr noundef captures(none) %0, i32 noundef %1) local_unnamed_addr {
; CHECK-LABEL: define dso_local void @func(ptr noundef captures(none) %0, i32 noundef %1) local_unnamed_addr {
; CHECK-NEXT:  [[CHECK:%.*]] = icmp slt i32 %1, 100
; CHECK-NEXT:  br i1 [[CHECK]], label [[LOOP_PREHEADER:%.*]], label [[LOOP_EXIT:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:  [[SEXT_START:%.*]] = sext i32 %1 to i64
; CHECK-NEXT:  br label [[LOOP_BODY:%.*]]
; CHECK:       loop.exit.loopexit:
; CHECK-NEXT:  br label [[LOOPEXIT:%.*]]
; CHECK:       loop.exit:
; CHECK-NEXT:  ret void
; CHECK:       loop.body:
; CHECK-NEXT:  [[IV:%.*]] = phi i64 [ [[POSTINC:%.*]], [[LOOP_BODY]] ], [ [[SEXT_START:%.*]], [[LOOP_PREHEADER]] ]
; CHECK-NEXT:  [[GEP:%.*]] = getelementptr inbounds i32, ptr %0, i64 [[IV]]
; CHECK-NEXT:  [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:  [[DATA:%.*]] = add nsw i32 [[LOAD]], 1
; CHECK-NEXT:  store i32 [[DATA]], ptr [[GEP]], align 4
; CHECK-NEXT:  [[POSTINC:%.*]] = add nsw i64 [[IV]], 1
; CHECK-NEXT:  [[LFTR_WIDEIV:%.*]] = trunc nsw i64 [[POSTINC]] to i32
; CHECK-NEXT:  [[EXITCOND:%.*]] = icmp ne i32 [[LFTR_WIDEIV]], 100
; CHECK-NEXT:  br i1 [[EXITCOND]], label [[LOOP_BODY]], label [[LOOP_EXIT_LOOPEXIT:%.*]]
; CHECK-NEXT:  }
  %3 = icmp slt i32 %1, 100
  br i1 %3, label %loop.preheader, label %loop.exit
loop.preheader:                                                
  br label %loop.body

loop.exit: 
  ret void
loop.body:                                                
  %8 = phi i32 [ %13, %loop.body ], [ %1, %loop.preheader ]
  %9 = sext i32 %8 to i64
  %10 = getelementptr inbounds i32, ptr %0, i64 %9
  %11 = load i32, ptr %10, align 4
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr %10, align 4
  %13 = add nsw i32 %8, 1
  %14 = icmp slt i32 %8, 99
  br i1 %14, label %loop.body, label %loop.exit
}
