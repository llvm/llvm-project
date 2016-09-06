; RUN: opt < %s -O3 | \
; RUN:   llc -mtriple=thumbv7-apple-darwin10 -mattr=+neon | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"

define void @fred(i32 %three_by_three, i8* %in, double %dt1, i32 %x_size, i32 %y_size, i8* %bp) nounwind {
entry:
; -- The loop following the load should only use a single add-literation
;    instruction.
; CHECK: vldr
; CHECK-NOT: adds
; CHECK: subsections_via_symbols


  %three_by_three_addr = alloca i32               ; <i32*> [#uses=2]
  %in_addr = alloca i8*                           ; <i8**> [#uses=2]
  %dt_addr = alloca float                         ; <float*> [#uses=4]
  %x_size_addr = alloca i32                       ; <i32*> [#uses=2]
  %y_size_addr = alloca i32                       ; <i32*> [#uses=1]
  %bp_addr = alloca i8*                           ; <i8**> [#uses=1]
  %tmp_image = alloca i8*                         ; <i8**> [#uses=0]
  %out = alloca i8*                               ; <i8**> [#uses=1]
  %cp = alloca i8*                                ; <i8**> [#uses=0]
  %dpt = alloca i8*                               ; <i8**> [#uses=4]
  %dp = alloca i8*                                ; <i8**> [#uses=2]
  %ip = alloca i8*                                ; <i8**> [#uses=0]
  %centre = alloca i32                            ; <i32*> [#uses=0]
  %tmp = alloca i32                               ; <i32*> [#uses=0]
  %brightness = alloca i32                        ; <i32*> [#uses=0]
  %area = alloca i32                              ; <i32*> [#uses=0]
  %y = alloca i32                                 ; <i32*> [#uses=0]
  %x = alloca i32                                 ; <i32*> [#uses=2]
  %j = alloca i32                                 ; <i32*> [#uses=6]
  %i = alloca i32                                 ; <i32*> [#uses=1]
  %mask_size = alloca i32                         ; <i32*> [#uses=5]
  %increment = alloca i32                         ; <i32*> [#uses=1]
  %n_max = alloca i32                             ; <i32*> [#uses=4]
  %temp = alloca float                            ; <float*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 %three_by_three, i32* %three_by_three_addr
  store i8* %in, i8** %in_addr
  %dt = fptrunc double %dt1 to float              ; <float> [#uses=1]
  store float %dt, float* %dt_addr
  store i32 %x_size, i32* %x_size_addr
  store i32 %y_size, i32* %y_size_addr
  store i8* %bp, i8** %bp_addr
  %0 = load i8*, i8** %in_addr, align 4                ; <i8*> [#uses=1]
  store i8* %0, i8** %out, align 4
  %1 = call  i32 (...) @foo() nounwind ; <i32> [#uses=1]
  store i32 %1, i32* %i, align 4
  %2 = load i32, i32* %three_by_three_addr, align 4    ; <i32> [#uses=1]
  %3 = icmp eq i32 %2, 0                          ; <i1> [#uses=1]
  br i1 %3, label %bb, label %bb2

bb:                                               ; preds = %entry
  %4 = load float, float* %dt_addr, align 4              ; <float> [#uses=1]
  %5 = fpext float %4 to double                   ; <double> [#uses=1]
  %6 = fmul double %5, 1.500000e+00               ; <double> [#uses=1]
  %7 = fptosi double %6 to i32                    ; <i32> [#uses=1]
  %8 = add nsw i32 %7, 1                          ; <i32> [#uses=1]
  store i32 %8, i32* %mask_size, align 4
  br label %bb3

bb2:                                              ; preds = %entry
  store i32 1, i32* %mask_size, align 4
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %9 = load i32, i32* %mask_size, align 4              ; <i32> [#uses=1]
  %10 = mul i32 %9, 2                             ; <i32> [#uses=1]
  %11 = add nsw i32 %10, 1                        ; <i32> [#uses=1]
  store i32 %11, i32* %n_max, align 4
  %12 = load i32, i32* %x_size_addr, align 4           ; <i32> [#uses=1]
  %13 = load i32, i32* %n_max, align 4                 ; <i32> [#uses=1]
  %14 = sub i32 %12, %13                          ; <i32> [#uses=1]
  store i32 %14, i32* %increment, align 4
  %15 = load i32, i32* %n_max, align 4                 ; <i32> [#uses=1]
  %16 = load i32, i32* %n_max, align 4                 ; <i32> [#uses=1]
  %17 = mul i32 %15, %16                          ; <i32> [#uses=1]
  %18 = call  noalias i8* @malloc(i32 %17) nounwind ; <i8*> [#uses=1]
  store i8* %18, i8** %dp, align 4
  %19 = load i8*, i8** %dp, align 4                    ; <i8*> [#uses=1]
  store i8* %19, i8** %dpt, align 4
  %20 = load float, float* %dt_addr, align 4             ; <float> [#uses=1]
  %21 = load float, float* %dt_addr, align 4             ; <float> [#uses=1]
  %22 = fmul float %20, %21                       ; <float> [#uses=1]
  %23 = fsub float -0.000000e+00, %22             ; <float> [#uses=1]
  store float %23, float* %temp, align 4
  %24 = load i32, i32* %mask_size, align 4             ; <i32> [#uses=1]
  %25 = sub i32 0, %24                            ; <i32> [#uses=1]
  store i32 %25, i32* %j, align 4
  br label %bb5

bb4:                                              ; preds = %bb5
  %26 = load i32, i32* %j, align 4                     ; <i32> [#uses=1]
  %27 = load i32, i32* %j, align 4                     ; <i32> [#uses=1]
  %28 = mul i32 %26, %27                          ; <i32> [#uses=1]
  %29 = sitofp i32 %28 to double                  ; <double> [#uses=1]
  %30 = fmul double %29, 1.234000e+00             ; <double> [#uses=1]
  %31 = fptosi double %30 to i32                  ; <i32> [#uses=1]
  store i32 %31, i32* %x, align 4
  %32 = load i32, i32* %x, align 4                     ; <i32> [#uses=1]
  %33 = trunc i32 %32 to i8                       ; <i8> [#uses=1]
  %34 = load i8*, i8** %dpt, align 4                   ; <i8*> [#uses=1]
  store i8 %33, i8* %34, align 1
  %35 = load i8*, i8** %dpt, align 4                   ; <i8*> [#uses=1]
  %36 = getelementptr inbounds i8, i8* %35, i64 1     ; <i8*> [#uses=1]
  store i8* %36, i8** %dpt, align 4
  %37 = load i32, i32* %j, align 4                     ; <i32> [#uses=1]
  %38 = add nsw i32 %37, 1                        ; <i32> [#uses=1]
  store i32 %38, i32* %j, align 4
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  %39 = load i32, i32* %j, align 4                     ; <i32> [#uses=1]
  %40 = load i32, i32* %mask_size, align 4             ; <i32> [#uses=1]
  %41 = icmp sle i32 %39, %40                     ; <i1> [#uses=1]
  br i1 %41, label %bb4, label %bb6

bb6:                                              ; preds = %bb5
  br label %return

return:                                           ; preds = %bb6
  ret void
}

declare i32 @foo(...)

declare noalias i8* @malloc(i32) nounwind
