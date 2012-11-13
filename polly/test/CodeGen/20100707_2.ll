; RUN: opt %loadPolly %defaultOpts -polly-codegen %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@win193 = external global [4 x [36 x double]], align 32 ; <[4 x [36 x double]]*> [#uses=3]
@sb_sample = external global [2 x [2 x [18 x [32 x double]]]], align 32 ; <[2 x [2 x [18 x [32 x double]]]]*> [#uses=2]

define void @mdct_sub48() nounwind {
entry:
  br i1 undef, label %bb, label %bb54

bb:                                               ; preds = %entry
  br label %bb54

bb3:                                              ; preds = %bb50
  br label %bb8

bb4:                                              ; preds = %bb8
  br label %bb8

bb8:                                              ; preds = %bb4, %bb3
  br i1 undef, label %bb4, label %bb9

bb9:                                              ; preds = %bb8
  br label %bb48

bb25:                                             ; preds = %bb48
  br i1 false, label %bb26, label %bb27

bb26:                                             ; preds = %bb48, %bb25
  br label %bb37

bb27:                                             ; preds = %bb25
  br i1 undef, label %bb32, label %bb35

bb32:                                             ; preds = %bb27
  br label %bb37

bb34:                                             ; preds = %bb35
  %0 = getelementptr inbounds [36 x double]* undef, i64 0, i64 0 ; <double*> [#uses=0]
  %1 = getelementptr inbounds [18 x [32 x double]]* undef, i64 0, i64 0 ; <[32 x double]*> [#uses=1]
  %2 = getelementptr inbounds [32 x double]* %1, i64 0, i64 0 ; <double*> [#uses=0]
  %3 = getelementptr inbounds [36 x double]* undef, i64 0, i64 0 ; <double*> [#uses=0]
  %4 = sub nsw i32 17, %k.4                       ; <i32> [#uses=1]
  %5 = getelementptr inbounds [2 x [2 x [18 x [32 x double]]]]* @sb_sample, i64 0, i64 0 ; <[2 x [18 x [32 x double]]]*> [#uses=1]
  %6 = getelementptr inbounds [2 x [18 x [32 x double]]]* %5, i64 0, i64 0 ; <[18 x [32 x double]]*> [#uses=1]
  %7 = sext i32 %4 to i64                         ; <i64> [#uses=1]
  %8 = getelementptr inbounds [18 x [32 x double]]* %6, i64 0, i64 %7 ; <[32 x double]*> [#uses=1]
  %9 = getelementptr inbounds [32 x double]* %8, i64 0, i64 0 ; <double*> [#uses=1]
  %10 = load double* %9, align 8                  ; <double> [#uses=0]
  %11 = fsub double 0.000000e+00, undef           ; <double> [#uses=1]
  %12 = getelementptr inbounds double* getelementptr inbounds ([4 x [36 x double]]* @win193, i64 0, i64 2, i64 4), i64 0 ; <double*> [#uses=1]
  store double %11, double* %12, align 8
  %13 = add nsw i32 %k.4, 9                       ; <i32> [#uses=1]
  %14 = add nsw i32 %k.4, 18                      ; <i32> [#uses=1]
  %15 = getelementptr inbounds [4 x [36 x double]]* @win193, i64 0, i64 0 ; <[36 x double]*> [#uses=1]
  %16 = sext i32 %14 to i64                       ; <i64> [#uses=1]
  %17 = getelementptr inbounds [36 x double]* %15, i64 0, i64 %16 ; <double*> [#uses=1]
  %18 = load double* %17, align 8                 ; <double> [#uses=0]
  %19 = sext i32 %k.4 to i64                      ; <i64> [#uses=1]
  %20 = getelementptr inbounds [18 x [32 x double]]* undef, i64 0, i64 %19 ; <[32 x double]*> [#uses=1]
  %21 = sext i32 %band.2 to i64                   ; <i64> [#uses=1]
  %22 = getelementptr inbounds [32 x double]* %20, i64 0, i64 %21 ; <double*> [#uses=1]
  %23 = load double* %22, align 8                 ; <double> [#uses=0]
  %24 = sext i32 %39 to i64                       ; <i64> [#uses=1]
  %25 = getelementptr inbounds [4 x [36 x double]]* @win193, i64 0, i64 %24 ; <[36 x double]*> [#uses=1]
  %26 = getelementptr inbounds [36 x double]* %25, i64 0, i64 0 ; <double*> [#uses=1]
  %27 = load double* %26, align 8                 ; <double> [#uses=0]
  %28 = sub nsw i32 17, %k.4                      ; <i32> [#uses=1]
  %29 = getelementptr inbounds [2 x [2 x [18 x [32 x double]]]]* @sb_sample, i64 0, i64 0 ; <[2 x [18 x [32 x double]]]*> [#uses=1]
  %30 = getelementptr inbounds [2 x [18 x [32 x double]]]* %29, i64 0, i64 0 ; <[18 x [32 x double]]*> [#uses=1]
  %31 = sext i32 %28 to i64                       ; <i64> [#uses=1]
  %32 = getelementptr inbounds [18 x [32 x double]]* %30, i64 0, i64 %31 ; <[32 x double]*> [#uses=1]
  %33 = getelementptr inbounds [32 x double]* %32, i64 0, i64 0 ; <double*> [#uses=1]
  %34 = load double* %33, align 8                 ; <double> [#uses=0]
  %35 = sext i32 %13 to i64                       ; <i64> [#uses=1]
  %36 = getelementptr inbounds double* getelementptr inbounds ([4 x [36 x double]]* @win193, i64 0, i64 2, i64 4), i64 %35 ; <double*> [#uses=1]
  store double 0.000000e+00, double* %36, align 8
  %37 = sub nsw i32 %k.4, 1                       ; <i32> [#uses=1]
  br label %bb35

bb35:                                             ; preds = %bb34, %bb27
  %k.4 = phi i32 [ %37, %bb34 ], [ 8, %bb27 ]     ; <i32> [#uses=6]
  br i1 undef, label %bb34, label %bb36

bb36:                                             ; preds = %bb35
  unreachable

bb37:                                             ; preds = %bb32, %bb26
  %38 = add nsw i32 %band.2, 1                    ; <i32> [#uses=1]
  br label %bb48

bb48:                                             ; preds = %bb37, %bb9
  %band.2 = phi i32 [ %38, %bb37 ], [ 0, %bb9 ]   ; <i32> [#uses=2]
  %39 = load i32* null, align 8                   ; <i32> [#uses=1]
  br i1 undef, label %bb26, label %bb25

bb50:                                             ; preds = %bb54
  br i1 undef, label %bb3, label %bb51

bb51:                                             ; preds = %bb50
  br i1 undef, label %bb52, label %bb53

bb52:                                             ; preds = %bb51
  unreachable

bb53:                                             ; preds = %bb51
  br label %bb54

bb54:                                             ; preds = %bb53, %bb, %entry
  br i1 undef, label %bb50, label %return

return:                                           ; preds = %bb54
  ret void
}
