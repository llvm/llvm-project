; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -analyze < %s

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_top_split
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_top_split[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_top_split[] -> [0, 0, 0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_top_split[] -> MemRef_25[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_top_split[] -> MemRef_26[] };
; CHECK-NEXT:     Stmt_L_4
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] : i0 >= 0 and i0 <= -1 + p_0 and i1 >= 0 and i1 <= -1 + p_0 and i2 >= 0 and i2 <= -1 + p_0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] -> [1, i0, i1, i2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] -> MemRef_25[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] -> MemRef_19[i1, i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] -> MemRef_5[i2, i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] -> MemRef_12[i2, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p_0, p_1, p_2] -> { Stmt_L_4[i0, i1, i2] -> MemRef_19[i1, i0] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%jl_value_t = type { %jl_value_t* }

define %jl_value_t* @julia_gemm_22583(%jl_value_t*, %jl_value_t**, i32) {
top:
  br label %top.split

top.split:                                        ; preds = %top
  %3 = load %jl_value_t*, %jl_value_t** %1, align 8
  %4 = bitcast %jl_value_t* %3 to double**
  %5 = load double*, double** %4, align 8
  %6 = getelementptr inbounds %jl_value_t, %jl_value_t* %3, i64 3, i32 0
  %7 = bitcast %jl_value_t** %6 to i64*
  %8 = load i64, i64* %7, align 8
  %9 = getelementptr %jl_value_t*, %jl_value_t** %1, i64 1
  %10 = load %jl_value_t*, %jl_value_t** %9, align 8
  %11 = bitcast %jl_value_t* %10 to double**
  %12 = load double*, double** %11, align 8
  %13 = getelementptr inbounds %jl_value_t, %jl_value_t* %10, i64 3, i32 0
  %14 = bitcast %jl_value_t** %13 to i64*
  %15 = load i64, i64* %14, align 8
  %16 = getelementptr %jl_value_t*, %jl_value_t** %1, i64 2
  %17 = load %jl_value_t*, %jl_value_t** %16, align 8
  %18 = bitcast %jl_value_t* %17 to double**
  %19 = load double*, double** %18, align 8
  %20 = getelementptr inbounds %jl_value_t, %jl_value_t* %17, i64 3, i32 0
  %21 = bitcast %jl_value_t** %20 to i64*
  %22 = load i64, i64* %21, align 8
  %23 = icmp sgt i64 %8, 0
  %24 = select i1 %23, i64 %8, i64 0
  %25 = add i64 %24, 1
  %26 = icmp eq i64 %24, 0
  br i1 %26, label %L.11, label %L.preheader

L.preheader:                                      ; preds = %top.split
  br label %L

L:                                                ; preds = %L.preheader, %L.9
  %"#s5.0" = phi i64 [ %27, %L.9 ], [ 1, %L.preheader ]
  %27 = add i64 %"#s5.0", 1
  br i1 %26, label %L.9, label %L.2.preheader

L.2.preheader:                                    ; preds = %L
  br label %L.2

L.2:                                              ; preds = %L.2.preheader, %L.7
  %"#s4.0" = phi i64 [ %28, %L.7 ], [ 1, %L.2.preheader ]
  %28 = add i64 %"#s4.0", 1
  br i1 %26, label %L.7, label %L.4.preheader

L.4.preheader:                                    ; preds = %L.2
  br label %L.4

L.4:                                              ; preds = %L.4.preheader, %L.4
  %"#s3.0" = phi i64 [ %29, %L.4 ], [ 1, %L.4.preheader ]
  %29 = add i64 %"#s3.0", 1
  %30 = add i64 %"#s5.0", -1
  %31 = add i64 %"#s4.0", -1
  %32 = mul i64 %31, %22
  %33 = add i64 %32, %30
  %34 = getelementptr double, double* %19, i64 %33
  %35 = load double, double* %34, align 8
  %36 = add i64 %"#s3.0", -1
  %37 = mul i64 %36, %8
  %38 = add i64 %37, %30
  %39 = getelementptr double, double* %5, i64 %38
  %40 = load double, double* %39, align 8
  %41 = mul i64 %36, %15
  %42 = add i64 %41, %31
  %43 = getelementptr double, double* %12, i64 %42
  %44 = load double, double* %43, align 8
  %45 = fmul double %40, %44
  %46 = fadd double %35, %45
  store double %46, double* %34, align 8
  %47 = icmp eq i64 %29, %25
  br i1 %47, label %L.7.loopexit, label %L.4

L.7.loopexit:                                     ; preds = %L.4
  br label %L.7

L.7:                                              ; preds = %L.7.loopexit, %L.2
  %48 = icmp eq i64 %28, %25
  br i1 %48, label %L.9.loopexit, label %L.2

L.9.loopexit:                                     ; preds = %L.7
  br label %L.9

L.9:                                              ; preds = %L.9.loopexit, %L
  %49 = icmp eq i64 %27, %25
  br i1 %49, label %L.11.loopexit, label %L

L.11.loopexit:                                    ; preds = %L.9
  br label %L.11

L.11:                                             ; preds = %L.11.loopexit, %top.split
  ret %jl_value_t* inttoptr (i64 140220477440016 to %jl_value_t*)
}
