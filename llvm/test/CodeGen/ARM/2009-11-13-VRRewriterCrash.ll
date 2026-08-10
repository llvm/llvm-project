; RUN: llc -mtriple=armv7-eabi -mcpu=cortex-a8 < %s
; PR5412
; rdar://7384107

%bar = type { %quad, float, float, [3 x ptr], [3 x ptr], [2 x ptr], [3 x i8], i8 }
%baz = type { ptr, i32 }
%foo = type { i8, %quux, %quad, float, [64 x %quuz], [128 x %bar], i32, %baz, %baz }
%quad = type { [4 x float] }
%quux = type { [4 x ptr], [4 x float], i32 }
%quuz = type { %quad, %quad }

define arm_aapcs_vfpcc ptr @aaa(ptr nocapture %this, ptr %a, ptr %b, ptr %c, i8 zeroext %forced) {
entry:
  %0 = load ptr, ptr undef, align 4                 ; <ptr> [#uses=2]
  br i1 false, label %bb85, label %bb

bb:                                               ; preds = %entry
  br i1 undef, label %bb3.i, label %bb2.i

bb2.i:                                            ; preds = %bb
  br label %bb3.i

bb3.i:                                            ; preds = %bb2.i, %bb
  %1 = getelementptr inbounds %quuz, ptr %a, i32 0, i32 1, i32 0, i32 0 ; <ptr> [#uses=1]
  %2 = fsub float 0.000000e+00, undef             ; <float> [#uses=1]
  %3 = getelementptr inbounds %quuz, ptr %b, i32 0, i32 1, i32 0, i32 1 ; <ptr> [#uses=1]
  %4 = getelementptr inbounds %quuz, ptr %b, i32 0, i32 1, i32 0, i32 2 ; <ptr> [#uses=1]
  %5 = fsub float 0.000000e+00, undef             ; <float> [#uses=1]
  %6 = getelementptr inbounds %quuz, ptr %c, i32 0, i32 1, i32 0, i32 0 ; <ptr> [#uses=1]
  %7 = getelementptr inbounds %quuz, ptr %c, i32 0, i32 1, i32 0, i32 1 ; <ptr> [#uses=1]
  %8 = fsub float undef, undef                    ; <float> [#uses=1]
  %9 = fmul float 0.000000e+00, %8                ; <float> [#uses=1]
  %10 = fmul float %5, 0.000000e+00               ; <float> [#uses=1]
  %11 = fsub float %9, %10                        ; <float> [#uses=3]
  %12 = fmul float %2, 0.000000e+00               ; <float> [#uses=1]
  %13 = fmul float 0.000000e+00, undef            ; <float> [#uses=1]
  %14 = fsub float %12, %13                       ; <float> [#uses=2]
  store float %14, ptr undef
  %15 = getelementptr inbounds %bar, ptr %0, i32 0, i32 0, i32 0, i32 3 ; <ptr> [#uses=1]
  store float 0.000000e+00, ptr %15
  %16 = fmul float %11, %11                       ; <float> [#uses=1]
  %17 = fadd float %16, 0.000000e+00              ; <float> [#uses=1]
  %18 = fadd float %17, undef                     ; <float> [#uses=1]
  %19 = call arm_aapcs_vfpcc  float @sqrtf(float %18) readnone ; <float> [#uses=2]
  %20 = fcmp ogt float %19, 0x3F1A36E2E0000000    ; <i1> [#uses=1]
  %21 = load float, ptr %1, align 4                   ; <float> [#uses=2]
  %22 = load float, ptr %3, align 4                   ; <float> [#uses=2]
  %23 = load float, ptr undef, align 4                ; <float> [#uses=2]
  %24 = load float, ptr %4, align 4                   ; <float> [#uses=2]
  %25 = fsub float %23, %24                       ; <float> [#uses=2]
  %26 = fmul float 0.000000e+00, %25              ; <float> [#uses=1]
  %27 = fsub float %26, undef                     ; <float> [#uses=1]
  %28 = fmul float %14, 0.000000e+00              ; <float> [#uses=1]
  %29 = fmul float %11, %25                       ; <float> [#uses=1]
  %30 = fsub float %28, %29                       ; <float> [#uses=1]
  %31 = fsub float undef, 0.000000e+00            ; <float> [#uses=1]
  %32 = fmul float %21, %27                       ; <float> [#uses=1]
  %33 = fmul float undef, %30                     ; <float> [#uses=1]
  %34 = fadd float %32, %33                       ; <float> [#uses=1]
  %35 = fmul float %23, %31                       ; <float> [#uses=1]
  %36 = fadd float %34, %35                       ; <float> [#uses=1]
  %37 = load float, ptr %6, align 4                   ; <float> [#uses=2]
  %38 = load float, ptr %7, align 4                   ; <float> [#uses=2]
  %39 = fsub float %22, %38                       ; <float> [#uses=2]
  %40 = load float, ptr undef, align 4                ; <float> [#uses=1]
  %41 = load float, ptr null, align 4                 ; <float> [#uses=2]
  %42 = fmul float %41, undef                     ; <float> [#uses=1]
  %43 = fmul float undef, %39                     ; <float> [#uses=1]
  %44 = fsub float %42, %43                       ; <float> [#uses=1]
  %45 = fmul float undef, %39                     ; <float> [#uses=1]
  %46 = fmul float %41, 0.000000e+00              ; <float> [#uses=1]
  %47 = fsub float %45, %46                       ; <float> [#uses=1]
  %48 = fmul float 0.000000e+00, %44              ; <float> [#uses=1]
  %49 = fmul float %22, undef                     ; <float> [#uses=1]
  %50 = fadd float %48, %49                       ; <float> [#uses=1]
  %51 = fmul float %24, %47                       ; <float> [#uses=1]
  %52 = fadd float %50, %51                       ; <float> [#uses=1]
  %53 = fsub float %37, %21                       ; <float> [#uses=2]
  %54 = fmul float undef, undef                   ; <float> [#uses=1]
  %55 = fmul float undef, undef                   ; <float> [#uses=1]
  %56 = fsub float %54, %55                       ; <float> [#uses=1]
  %57 = fmul float undef, %53                     ; <float> [#uses=1]
  %58 = load float, ptr undef, align 4                ; <float> [#uses=2]
  %59 = fmul float %58, undef                     ; <float> [#uses=1]
  %60 = fsub float %57, %59                       ; <float> [#uses=1]
  %61 = fmul float %58, undef                     ; <float> [#uses=1]
  %62 = fmul float undef, %53                     ; <float> [#uses=1]
  %63 = fsub float %61, %62                       ; <float> [#uses=1]
  %64 = fmul float %37, %56                       ; <float> [#uses=1]
  %65 = fmul float %38, %60                       ; <float> [#uses=1]
  %66 = fadd float %64, %65                       ; <float> [#uses=1]
  %67 = fmul float %40, %63                       ; <float> [#uses=1]
  %68 = fadd float %66, %67                       ; <float> [#uses=1]
  %69 = select i1 undef, float %36, float %52     ; <float> [#uses=1]
  %70 = select i1 undef, float %69, float %68     ; <float> [#uses=1]
  %iftmp.164.0 = select i1 %20, float %19, float 1.000000e+00 ; <float> [#uses=1]
  %71 = fdiv float %70, %iftmp.164.0              ; <float> [#uses=1]
  store float %71, ptr null, align 4
  %72 = icmp eq ptr null, %0                    ; <i1> [#uses=1]
  br i1 %72, label %bb4.i97, label %ccc.exit98

bb4.i97:                                          ; preds = %bb3.i
  %73 = load ptr, ptr undef, align 4                ; <ptr> [#uses=0]
  br label %ccc.exit98

ccc.exit98:                                       ; preds = %bb4.i97, %bb3.i
  ret ptr null

bb85:                                             ; preds = %entry
  ret ptr null
}

declare arm_aapcs_vfpcc float @sqrtf(float) readnone
