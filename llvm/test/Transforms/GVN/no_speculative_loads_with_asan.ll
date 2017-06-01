; RUN: opt -O3 -S %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
declare noalias i8* @_Znam(i64) #1

define i32 @TestNoAsan() {
  %1 = tail call noalias i8* @_Znam(i64 2)
  %2 = getelementptr inbounds i8, i8* %1, i64 1
  store i8 0, i8* %2, align 1
  store i8 0, i8* %1, align 1
  %3 = bitcast i8* %1 to i16*
  %4 = load i16, i16* %3, align 4
  %5 = icmp eq i16 %4, 0
  br i1 %5, label %11, label %6

; <label>:6                                       ; preds = %0
  %7 = getelementptr inbounds i8, i8* %1, i64 2
  %8 = bitcast i8* %7 to i16*
  %9 = load i16, i16* %8, align 2
  %10 = sext i16 %9 to i32
  br label %11

; <label>:11                                      ; preds = %0, %6
  %12 = phi i32 [ %10, %6 ], [ 0, %0 ]
  ret i32 %12
}

; CHECK-LABEL: @TestNoAsan
; CHECK: ret i32 0

define i32 @TestAsan() sanitize_address {
  %1 = tail call noalias i8* @_Znam(i64 2)
  %2 = getelementptr inbounds i8, i8* %1, i64 1
  store i8 0, i8* %2, align 1
  store i8 0, i8* %1, align 1
  %3 = bitcast i8* %1 to i16*
  %4 = load i16, i16* %3, align 4
  %5 = icmp eq i16 %4, 0
  br i1 %5, label %11, label %6

; <label>:6                                       ; preds = %0
  %7 = getelementptr inbounds i8, i8* %1, i64 2
  %8 = bitcast i8* %7 to i16*
  %9 = load i16, i16* %8, align 2
  %10 = sext i16 %9 to i32
  br label %11

; <label>:11                                      ; preds = %0, %6
  %12 = phi i32 [ %10, %6 ], [ 0, %0 ]
  ret i32 %12
}

; CHECK-LABEL: @TestAsan
; CHECK-NOT: %[[LOAD:[^ ]+]] = load i32
; CHECK: {{.*}} = phi

