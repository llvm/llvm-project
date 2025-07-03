; Test indvars for eliminating the Pure LoopCounter.
; RUN: opt -S -passes=indvars -enable-pure-loop-counter-elimination < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @pure_loop_counter_elimination(ptr nocapture readonly %0, ptr nocapture readonly %1, ptr nocapture readonly %2)  {
; CHECK-LABEL: define void @pure_loop_counter_elimination(ptr readonly captures(none) %0, ptr readonly captures(none) %1, ptr readonly captures(none) %2) {
; CHECK-NEXT:  [[entry:.*]]:
; CHECK-NEXT:  [[IS:%.*]] = load i32, ptr %1, align 4
; CHECK-NEXT:  [[IS_SEXT:%.*]] = sext i32 [[IS]] to i64
; CHECK-NEXT:  [[IE:%.*]] = load i32, ptr %2, align 4
; CHECK-NEXT:  [[IE_SEXT:%.*]] = sext i32 [[IE]] to i64
; CHECK-NEXT:  [[IE_IS:%.*]] = sub i64 [[IE_SEXT]], [[IS_SEXT]]
; CHECK-NEXT:  [[IE_IS_1:%.*]] = add nsw i64 [[IE_IS]], 1
; CHECK-NEXT:  br label %preheader

; CHECK:       [[preheader:.*]]:                                        
; CHECK-NEXT:  [[Check:%.*]] = icmp sgt i64 [[IE_SEXT]], [[IS_SEXT]]
; CHECK-NEXT:  br i1 [[Check]], label %loop.preheader, label %end

; CHECK:       [[loop_preheader:.*]]:                            
; CHECK-NEXT:  [[IE_Plus_2:%.*]] = add i64 [[IE_SEXT]], 2
; CHECK-NEXT:  [[MIN:%.*]] = call i64 @llvm.umin.i64(i64 [[IE_IS_1]], i64 1)
; CHECK-NEXT:  [[TEMP:%.*]] = sub i64 [[IE_Plus_2]], [[MIN]]
; CHECK-NEXT:  br label %loop

; CHECK:       [[loop:.*]]:                                             
; CHECK-NEXT:  [[IV:%.*]] = phi i64 [ %15, %loop ], [ [[IS_SEXT]], %loop.preheader ]
; CHECK-NEXT:  [[ADDR:%.*]] = getelementptr float, ptr %0, i64 [[IV]]
; CHECK-NEXT:  [[DATA:%.*]] = load float, ptr [[ADDR]], align 4
; CHECK-NEXT:  [[DATA_PUS_1:%.*]] = fadd fast float [[DATA]], 1.000000e+00
; CHECK-NEXT:  store float [[DATA_PUS_1]], ptr [[ADDR]], align 4
; CHECK-NEXT:  [[PostAdd:%.*]] = add i64 [[IV]], 1
; CHECK-NEXT:  [[ExitCond:%.*]] = icmp ne i64 [[PostAdd]], [[TEMP]]
; CHECK-NEXT:  br i1 [[ExitCond]], label %loop, label %end.loopexit

; CHECK:       [[end_loopexit:.*]]:                                   
; CHECK-NEXT:  br label %end

; CHECK:       [[end:.*]]:                                              
; CHECK-NEXT:   ret void
; CHECK-LABEL: }

entry: 
  %3 = load i32, ptr %1, align 4
  %4 = sext i32 %3 to i64
  %5 = load i32, ptr %2, align 4
  %6 = sext i32 %5 to i64
  %7 = sub nsw i64 %6, %4
  %8 = add nsw i64 %7, 1
  br label %preheader

preheader:
  %cmp = icmp sgt i64 %6, %4
  br i1 %cmp, label %loop, label %end

loop:                                           
  %9  = phi i64 [ %15, %loop ], [ %8, %preheader ]
  %10 = phi i64 [ %14, %loop ], [ %4, %preheader ]
  %11 = getelementptr float, ptr %0, i64 %10
  %12 = load float, ptr %11, align 4
  %13 = fadd fast float %12, 1.000000e+00
  store float %13, ptr %11, align 4
  %14 = add i64 %10, 1
  %15 = add nsw i64 %9, -1
  %16 = icmp ugt i64 %9, 1
  br i1 %16, label %loop, label %end

end:                                      
  ret void
}