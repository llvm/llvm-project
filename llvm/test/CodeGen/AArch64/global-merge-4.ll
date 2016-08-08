; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-enable-global-merge -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"
target triple = "arm64-apple-ios7.0.0"

@bar = internal global [5 x i32] zeroinitializer, align 4
@baz = internal global [5 x i32] zeroinitializer, align 4
@foo = internal global [5 x i32] zeroinitializer, align 4

; Function Attrs: nounwind ssp
define internal void @initialize() #0 {
  %1 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %1, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 0), align 4
  %2 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %2, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 0), align 4
  %3 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %3, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 1), align 4
  %4 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %4, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 1), align 4
  %5 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %5, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 2), align 4
  %6 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %6, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 2), align 4
  %7 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %7, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 3), align 4
  %8 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %8, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 3), align 4
  %9 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %9, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 4), align 4
  %10 = tail call i32 bitcast (i32 (...)* @calc to i32 ()*)() #2
  store i32 %10, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 4), align 4
  ret void
}

declare i32 @calc(...)

; Function Attrs: nounwind ssp
define internal void @calculate() #0 {
  %1 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 0), align 4
  %2 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 0), align 4
  %3 = mul nsw i32 %2, %1
  store i32 %3, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo, i64 0, i64 0), align 4
  %4 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 1), align 4
  %5 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 1), align 4
  %6 = mul nsw i32 %5, %4
  store i32 %6, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo, i64 0, i64 1), align 4
  %7 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 2), align 4
  %8 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 2), align 4
  %9 = mul nsw i32 %8, %7
  store i32 %9, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo, i64 0, i64 2), align 4
  %10 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 3), align 4
  %11 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 3), align 4
  %12 = mul nsw i32 %11, %10
  store i32 %12, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo, i64 0, i64 3), align 4
  %13 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @bar, i64 0, i64 4), align 4
  %14 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @baz, i64 0, i64 4), align 4
  %15 = mul nsw i32 %14, %13
  store i32 %15, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo, i64 0, i64 4), align 4
  ret void
}

; Function Attrs: nounwind readnone ssp
define internal i32* @returnFoo() #1 {
  ret i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo, i64 0, i64 0)
}

;CHECK:	.type	.L_MergedGlobals,@object  // @_MergedGlobals
;CHECK:	.local	.L_MergedGlobals
;CHECK:	.comm	.L_MergedGlobals,60,16

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone ssp }
attributes #2 = { nounwind }
