; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; Exercise getConstantEvolvingPHIOperands on an interesting loop.
; This should complete in milliseconds, not minutes.

; Just check that it actually ran trip count analysis.
; CHECK: Determining loop execution counts for: @test
define void @test() nounwind {
entry:
  br label %loop

loop:
  %iv = phi i32 [ %30, %loop ], [ 0, %entry ]
  %0 = add i32 %iv, 1
  %1 = add i32 %0, 2
  %2 = add i32 %1, %0
  %3 = add i32 %2, %1
  %4 = add i32 %3, %2
  %5 = add i32 %4, %3
  %6 = add i32 %5, %4
  %7 = add i32 %6, %5
  %8 = add i32 %7, %6
  %9 = add i32 %8, %7
  %10 = add i32 %9, %8
  %11 = add i32 %10, %9
  %12 = add i32 %11, %10
  %13 = add i32 %12, %11
  %14 = add i32 %13, %12
  %15 = add i32 %14, %13
  %16 = add i32 %15, %14
  %17 = add i32 %16, %15
  %18 = add i32 %17, %16
  %19 = add i32 %18, %17
  %20 = add i32 %19, %18
  %21 = add i32 %20, %19
  %22 = add i32 %21, %20
  %23 = add i32 %22, %21
  %24 = add i32 %23, %22
  %25 = add i32 %24, %23
  %26 = add i32 %25, %24
  %27 = add i32 %26, %25
  %28 = add i32 %27, %26
  %29 = add i32 %28, %27
  %30 = add i32 %29, %28
  %cmp = icmp eq i32 %30, -108
  br i1 %cmp, label %exit, label %loop

exit:
  unreachable
}
