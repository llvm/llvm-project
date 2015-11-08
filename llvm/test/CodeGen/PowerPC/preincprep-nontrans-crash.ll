; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64-i64:64-n32:64"
target triple = "powerpc64le-linux"

%struct.BSS1.0.9.28.39.43.46.47.54.56.57.64.65.69.71.144 = type <{ [220 x i8] }>

@.BSS1 = external unnamed_addr global %struct.BSS1.0.9.28.39.43.46.47.54.56.57.64.65.69.71.144, align 32

; Function Attrs: noinline nounwind
define void @ety2_() #0 {

; This test case used to crash because the preinc prep pass would assume that
; if X-Y could be simplified to a constant, than so could Y-X. While not
; desirable, we cannot actually make this guarantee.
; CHECK-LABEL: @ety2_

L.entry:
  %0 = load i32, i32* undef, align 4
  %1 = sext i32 %0 to i64
  %2 = shl nsw i64 %1, 3
  %3 = add nsw i64 %2, 8
  br label %L.LB1_425

L.LB1_425:                                        ; preds = %L.LB1_427, %L.entry
  %4 = phi i64 [ %21, %L.LB1_427 ], [ undef, %L.entry ]
  br i1 undef, label %L.LB1_427, label %L.LB1_816

L.LB1_816:                                        ; preds = %L.LB1_425
  switch i32 undef, label %L.LB1_432 [
    i32 30, label %L.LB1_805
    i32 10, label %L.LB1_451
    i32 20, label %L.LB1_451
  ]

L.LB1_451:                                        ; preds = %L.LB1_816, %L.LB1_816
  unreachable

L.LB1_432:                                        ; preds = %L.LB1_816
  %.in.31 = lshr i64 %4, 32
  %5 = trunc i64 %.in.31 to i32
  br i1 undef, label %L.LB1_769, label %L.LB1_455

L.LB1_455:                                        ; preds = %L.LB1_432
  unreachable

L.LB1_769:                                        ; preds = %L.LB1_432
  %6 = sext i32 %5 to i64
  %7 = add nsw i64 %6, 2
  %8 = add nsw i64 %6, -1
  %9 = mul i64 %8, %1
  %10 = add i64 %9, %7
  %11 = shl i64 %10, 3
  %12 = getelementptr i8, i8* undef, i64 %11
  %13 = mul nsw i64 %6, %1
  %14 = add i64 %7, %13
  %15 = shl i64 %14, 3
  %16 = getelementptr i8, i8* undef, i64 %15
  br i1 undef, label %L.LB1_662, label %L.LB1_662.prol

L.LB1_662.prol:                                   ; preds = %L.LB1_662.prol, %L.LB1_769
  %indvars.iv.next20.prol = add nuw nsw i64 undef, 1
  br i1 undef, label %L.LB1_662, label %L.LB1_662.prol

L.LB1_662:                                        ; preds = %L.LB1_437.2, %L.LB1_662.prol, %L.LB1_769
  %indvars.iv19 = phi i64 [ %indvars.iv.next20.3, %L.LB1_437.2 ], [ 0, %L.LB1_769 ], [ %indvars.iv.next20.prol, %L.LB1_662.prol ]
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %17 = mul i64 %indvars.iv.next20, %3
  %18 = getelementptr i8, i8* %16, i64 %17
  %19 = bitcast i8* %18 to double*
  store double 0.000000e+00, double* %19, align 8
  %indvars.iv.next20.1 = add nsw i64 %indvars.iv19, 2
  %20 = mul i64 %indvars.iv.next20.1, %3
  br i1 undef, label %L.LB1_437.2, label %L.LB1_824.2

L.LB1_427:                                        ; preds = %L.LB1_425
  %21 = load i64, i64* bitcast (i8* getelementptr inbounds (%struct.BSS1.0.9.28.39.43.46.47.54.56.57.64.65.69.71.144, %struct.BSS1.0.9.28.39.43.46.47.54.56.57.64.65.69.71.144* @.BSS1, i64 0, i32 0, i64 8) to i64*), align 8
  br label %L.LB1_425

L.LB1_805:                                        ; preds = %L.LB1_816
  ret void

L.LB1_824.2:                                      ; preds = %L.LB1_662
  %22 = getelementptr i8, i8* %12, i64 %20
  %23 = bitcast i8* %22 to double*
  store double 0.000000e+00, double* %23, align 8
  br label %L.LB1_437.2

L.LB1_437.2:                                      ; preds = %L.LB1_824.2, %L.LB1_662
  %indvars.iv.next20.3 = add nsw i64 %indvars.iv19, 4
  br label %L.LB1_662
}

attributes #0 = { noinline nounwind }

