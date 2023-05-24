; RUN: opt -passes="ipsccp<func-spec>,deadargelim" -force-specialization -S < %s | FileCheck %s --check-prefixes=COMMON,ITERS1
; RUN: opt -passes="ipsccp<func-spec>,deadargelim" -funcspec-max-iters=1 -force-specialization -S < %s | FileCheck %s --check-prefixes=COMMON,ITERS1
; RUN: opt -passes="ipsccp<func-spec>,deadargelim" -funcspec-max-iters=2 -force-specialization -S < %s | FileCheck %s --check-prefixes=COMMON,ITERS2
; RUN: opt -passes="ipsccp<func-spec>,deadargelim" -funcspec-max-iters=0 -force-specialization -S < %s | FileCheck %s --check-prefix=DISABLED

; DISABLED-NOT: @func.1(
; DISABLED-NOT: @func.2(
; DISABLED-NOT: @func.3(

define internal i32 @func(ptr %0, i32 %1, ptr nocapture %2) {
  %4 = alloca i32, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %4, align 4
  %6 = icmp slt i32 %5, 1
  br i1 %6, label %14, label %7

7:                                                ; preds = %3
  %8 = load i32, ptr %4, align 4
  %9 = sext i32 %8 to i64
  %10 = getelementptr inbounds i32, ptr %0, i64 %9
  call void %2(ptr %10)
  %11 = load i32, ptr %4, align 4
  %12 = add nsw i32 %11, -1
  %13 = call i32 @func(ptr %0, i32 %12, ptr %2)
  br label %14

14:                                               ; preds = %3, %7
  ret i32 0
}

define internal void @increment(ptr nocapture %0) {
  %2 = load i32, ptr %0, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr %0, align 4
  ret void
}

define internal void @decrement(ptr nocapture %0) {
  %2 = load i32, ptr %0, align 4
  %3 = add nsw i32 %2, -1
  store i32 %3, ptr %0, align 4
  ret void
}

define i32 @main(ptr %0, i32 %1) {
; COMMON:      define i32 @main(
; COMMON-NEXT:    call void @func.2(ptr [[TMP0:%.*]], i32 [[TMP1:%.*]])
; COMMON-NEXT:    call void @func.1(ptr [[TMP0]])
; COMMON-NEXT:    ret i32 0
;
  %3 = call i32 @func(ptr %0, i32 %1, ptr nonnull @increment)
  %4 = call i32 @func(ptr %0, i32 %3, ptr nonnull @decrement)
  ret i32 %4
}

; COMMON:      define internal void @func.1(
; COMMON-NEXT:    [[TMP2:%.*]] = alloca i32, align 4
; COMMON-NEXT:    store i32 0, ptr [[TMP2]], align 4
; COMMON-NEXT:    [[TMP3:%.*]] = load i32, ptr [[TMP2]], align 4
; COMMON-NEXT:    [[TMP4:%.*]] = icmp slt i32 [[TMP3]], 1
; COMMON-NEXT:    br i1 [[TMP4]], label [[TMP11:%.*]], label [[TMP5:%.*]]
; COMMON:      5:
; COMMON-NEXT:    [[TMP6:%.*]] = load i32, ptr [[TMP2]], align 4
; COMMON-NEXT:    [[TMP7:%.*]] = sext i32 [[TMP6]] to i64
; COMMON-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[TMP0:%.*]], i64 [[TMP7]]
; COMMON-NEXT:    call void @decrement(ptr [[TMP8]])
; COMMON-NEXT:    [[TMP9:%.*]] = load i32, ptr [[TMP2]], align 4
; COMMON-NEXT:    [[TMP10:%.*]] = add nsw i32 [[TMP9]], -1
; ITERS1-NEXT:    call void @func(ptr [[TMP0]], i32 [[TMP10]], ptr @decrement)
; ITERS2-NEXT:    call void @func.3(ptr [[TMP0]], i32 [[TMP10]])
; COMMON-NEXT:    br label [[TMP11:%.*]]
; COMMON:      11:
; COMMON-NEXT:    ret void
;
; COMMON:      define internal void @func.2(
; COMMON-NEXT:    [[TMP3:%.*]] = alloca i32, align 4
; COMMON-NEXT:    store i32 [[TMP1:%.*]], ptr [[TMP3]], align 4
; COMMON-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TMP3]], align 4
; COMMON-NEXT:    [[TMP5:%.*]] = icmp slt i32 [[TMP4]], 1
; COMMON-NEXT:    br i1 [[TMP5]], label [[TMP13:%.*]], label [[TMP6:%.*]]
; COMMON:      6:
; COMMON-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP3]], align 4
; COMMON-NEXT:    [[TMP8:%.*]] = sext i32 [[TMP7]] to i64
; COMMON-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[TMP0:%.*]], i64 [[TMP8]]
; COMMON-NEXT:    call void @increment(ptr [[TMP9]])
; COMMON-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP3]], align 4
; COMMON-NEXT:    [[TMP11:%.*]] = add nsw i32 [[TMP10]], -1
; COMMON-NEXT:    call void @func.2(ptr [[TMP0]], i32 [[TMP11]])
; COMMON-NEXT:    br label [[TMP12:%.*]]
; COMMON:      12:
; COMMON-NEXT:    ret void
;
; ITERS2:      define internal void @func.3(
; ITERS2-NEXT:    [[TMP3:%.*]] = alloca i32, align 4
; ITERS2-NEXT:    store i32 [[TMP1:%.*]], ptr [[TMP3]], align 4
; ITERS2-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TMP3]], align 4
; ITERS2-NEXT:    [[TMP5:%.*]] = icmp slt i32 [[TMP4]], 1
; ITERS2-NEXT:    br i1 [[TMP5]], label [[TMP13:%.*]], label [[TMP6:%.*]]
; ITERS2:      6:
; ITERS2-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP3]], align 4
; ITERS2-NEXT:    [[TMP8:%.*]] = sext i32 [[TMP7]] to i64
; ITERS2-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[TMP0:%.*]], i64 [[TMP8]]
; ITERS2-NEXT:    call void @decrement(ptr [[TMP9]])
; ITERS2-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP3]], align 4
; ITERS2-NEXT:    [[TMP11:%.*]] = add nsw i32 [[TMP10]], -1
; ITERS2-NEXT:    call void @func.3(ptr [[TMP0]], i32 [[TMP11]])
; ITERS2-NEXT:    br label [[TMP12:%.*]]
; ITERS2:      12:
; ITERS2-NEXT:    ret void

