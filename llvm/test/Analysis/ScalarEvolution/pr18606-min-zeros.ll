; RUN: opt -S -passes=indvars < %s | FileCheck %s

; CHECK: @test
; CHECK: %5 = add i32 %local_6_, %local_0_
; CHECK: %37 = mul i32 %36, %36

define i32 @test(i32, i32) {
bci_0:
  br label %bci_30

bci_68:                                           ; preds = %bci_45
  %local_6_.lcssa = phi i32 [ %local_6_, %bci_45 ]
  %.lcssa1.lcssa = phi i32 [ %37, %bci_45 ]
  %.lcssa.lcssa = phi i32 [ 34, %bci_45 ]
  %2 = add i32 %local_6_.lcssa, 262
  %3 = add i32 %2, %.lcssa1.lcssa
  %4 = add i32 %3, %.lcssa.lcssa
  ret i32 %4

bci_30:                                           ; preds = %bci_45, %bci_0
  %local_0_ = phi i32 [ %0, %bci_0 ], [ %38, %bci_45 ]
  %local_6_ = phi i32 [ 2, %bci_0 ], [ %39, %bci_45 ]
  %5 = add i32 %local_6_, %local_0_
  br label %bci_45

bci_45:                                           ; preds = %bci_30
  %6 = mul i32 %5, %5
  %7 = mul i32 %6, %6
  %8 = mul i32 %7, %7
  %9 = mul i32 %8, %8
  %10 = mul i32 %9, %9
  %11 = mul i32 %10, %10
  %12 = mul i32 %11, %11
  %13 = mul i32 %12, %12
  %14 = mul i32 %13, %13
  %15 = mul i32 %14, %14
  %16 = mul i32 %15, %15
  %17 = mul i32 %16, %16
  %18 = mul i32 %17, %17
  %19 = mul i32 %18, %18
  %20 = mul i32 %19, %19
  %21 = mul i32 %20, %20
  %22 = mul i32 %21, %21
  %23 = mul i32 %22, %22
  %24 = mul i32 %23, %23
  %25 = mul i32 %24, %24
  %26 = mul i32 %25, %25
  %27 = mul i32 %26, %26
  %28 = mul i32 %27, %27
  %29 = mul i32 %28, %28
  %30 = mul i32 %29, %29
  %31 = mul i32 %30, %30
  %32 = mul i32 %31, %31
  %33 = mul i32 %32, %32
  %34 = mul i32 %33, %33
  %35 = mul i32 %34, %34
  %36 = mul i32 %35, %35
  %37 = mul i32 %36, %36
  %38 = add i32 %37, -11
  %39 = add i32 %local_6_, 1
  %40 = icmp sgt i32 %39, 76
  br i1 %40, label %bci_68, label %bci_30
}
