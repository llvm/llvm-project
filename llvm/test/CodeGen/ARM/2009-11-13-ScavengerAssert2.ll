; RUN: llc -mtriple=armv7-eabi -mcpu=cortex-a8 < %s
; PR5412

%bar = type { %quad, float, float, [3 x %quuz*], [3 x %bar*], [2 x %bar*], [3 x i8], i8 }
%baz = type { %bar*, i32 }
%foo = type { i8, %quux, %quad, float, [64 x %quuz], [128 x %bar], i32, %baz, %baz }
%quad = type { [4 x float] }
%quux = type { [4 x %quuz*], [4 x float], i32 }
%quuz = type { %quad, %quad }

define arm_aapcs_vfpcc %bar* @aaa(%foo* nocapture %this, %quuz* %a, %quuz* %b, %quuz* %c, i8 zeroext %forced) {
entry:
  br i1 undef, label %bb85, label %bb

bb:                                               ; preds = %entry
  br i1 undef, label %bb3.i, label %bb2.i

bb2.i:                                            ; preds = %bb
  br label %bb3.i

bb3.i:                                            ; preds = %bb2.i, %bb
  %0 = getelementptr inbounds %quuz* %a, i32 0, i32 1, i32 0, i32 0 ; <float*> [#uses=0]
  %1 = fsub float 0.000000e+00, undef             ; <float> [#uses=1]
  %2 = getelementptr inbounds %quuz* %b, i32 0, i32 1, i32 0, i32 1 ; <float*> [#uses=2]
  %3 = load float* %2, align 4                    ; <float> [#uses=1]
  %4 = getelementptr inbounds %quuz* %a, i32 0, i32 1, i32 0, i32 1 ; <float*> [#uses=1]
  %5 = fsub float %3, undef                       ; <float> [#uses=2]
  %6 = getelementptr inbounds %quuz* %b, i32 0, i32 1, i32 0, i32 2 ; <float*> [#uses=2]
  %7 = load float* %6, align 4                    ; <float> [#uses=1]
  %8 = fsub float %7, undef                       ; <float> [#uses=1]
  %9 = getelementptr inbounds %quuz* %c, i32 0, i32 1, i32 0, i32 0 ; <float*> [#uses=2]
  %10 = load float* %9, align 4                   ; <float> [#uses=1]
  %11 = fsub float %10, undef                     ; <float> [#uses=2]
  %12 = getelementptr inbounds %quuz* %c, i32 0, i32 1, i32 0, i32 1 ; <float*> [#uses=2]
  %13 = load float* %12, align 4                  ; <float> [#uses=1]
  %14 = fsub float %13, undef                     ; <float> [#uses=1]
  %15 = load float* undef, align 4                ; <float> [#uses=1]
  %16 = fsub float %15, undef                     ; <float> [#uses=1]
  %17 = fmul float %5, %16                        ; <float> [#uses=1]
  %18 = fsub float %17, 0.000000e+00              ; <float> [#uses=5]
  %19 = fmul float %8, %11                        ; <float> [#uses=1]
  %20 = fsub float %19, undef                     ; <float> [#uses=3]
  %21 = fmul float %1, %14                        ; <float> [#uses=1]
  %22 = fmul float %5, %11                        ; <float> [#uses=1]
  %23 = fsub float %21, %22                       ; <float> [#uses=2]
  store float %18, float* undef
  %24 = getelementptr inbounds %bar* null, i32 0, i32 0, i32 0, i32 1 ; <float*> [#uses=2]
  store float %20, float* %24
  store float %23, float* undef
  %25 = getelementptr inbounds %bar* null, i32 0, i32 0, i32 0, i32 3 ; <float*> [#uses=0]
  %26 = fmul float %18, %18                       ; <float> [#uses=1]
  %27 = fadd float %26, undef                     ; <float> [#uses=1]
  %28 = fadd float %27, undef                     ; <float> [#uses=1]
  %29 = call arm_aapcs_vfpcc  float @sqrtf(float %28) readnone ; <float> [#uses=1]
  %30 = load float* null, align 4                 ; <float> [#uses=2]
  %31 = load float* %4, align 4                   ; <float> [#uses=2]
  %32 = load float* %2, align 4                   ; <float> [#uses=2]
  %33 = load float* null, align 4                 ; <float> [#uses=3]
  %34 = load float* %6, align 4                   ; <float> [#uses=2]
  %35 = fsub float %33, %34                       ; <float> [#uses=2]
  %36 = fmul float %20, %35                       ; <float> [#uses=1]
  %37 = fsub float %36, undef                     ; <float> [#uses=1]
  %38 = fmul float %23, 0.000000e+00              ; <float> [#uses=1]
  %39 = fmul float %18, %35                       ; <float> [#uses=1]
  %40 = fsub float %38, %39                       ; <float> [#uses=1]
  %41 = fmul float %18, 0.000000e+00              ; <float> [#uses=1]
  %42 = fmul float %20, 0.000000e+00              ; <float> [#uses=1]
  %43 = fsub float %41, %42                       ; <float> [#uses=1]
  %44 = fmul float 0.000000e+00, %37              ; <float> [#uses=1]
  %45 = fmul float %31, %40                       ; <float> [#uses=1]
  %46 = fadd float %44, %45                       ; <float> [#uses=1]
  %47 = fmul float %33, %43                       ; <float> [#uses=1]
  %48 = fadd float %46, %47                       ; <float> [#uses=2]
  %49 = load float* %9, align 4                   ; <float> [#uses=2]
  %50 = fsub float %30, %49                       ; <float> [#uses=1]
  %51 = load float* %12, align 4                  ; <float> [#uses=3]
  %52 = fsub float %32, %51                       ; <float> [#uses=2]
  %53 = load float* undef, align 4                ; <float> [#uses=2]
  %54 = load float* %24, align 4                  ; <float> [#uses=2]
  %55 = fmul float %54, undef                     ; <float> [#uses=1]
  %56 = fmul float undef, %52                     ; <float> [#uses=1]
  %57 = fsub float %55, %56                       ; <float> [#uses=1]
  %58 = fmul float undef, %52                     ; <float> [#uses=1]
  %59 = fmul float %54, %50                       ; <float> [#uses=1]
  %60 = fsub float %58, %59                       ; <float> [#uses=1]
  %61 = fmul float %30, %57                       ; <float> [#uses=1]
  %62 = fmul float %32, 0.000000e+00              ; <float> [#uses=1]
  %63 = fadd float %61, %62                       ; <float> [#uses=1]
  %64 = fmul float %34, %60                       ; <float> [#uses=1]
  %65 = fadd float %63, %64                       ; <float> [#uses=2]
  %66 = fcmp olt float %48, %65                   ; <i1> [#uses=1]
  %67 = fsub float %49, 0.000000e+00              ; <float> [#uses=1]
  %68 = fsub float %51, %31                       ; <float> [#uses=1]
  %69 = fsub float %53, %33                       ; <float> [#uses=1]
  %70 = fmul float undef, %67                     ; <float> [#uses=1]
  %71 = load float* undef, align 4                ; <float> [#uses=2]
  %72 = fmul float %71, %69                       ; <float> [#uses=1]
  %73 = fsub float %70, %72                       ; <float> [#uses=1]
  %74 = fmul float %71, %68                       ; <float> [#uses=1]
  %75 = fsub float %74, 0.000000e+00              ; <float> [#uses=1]
  %76 = fmul float %51, %73                       ; <float> [#uses=1]
  %77 = fadd float undef, %76                     ; <float> [#uses=1]
  %78 = fmul float %53, %75                       ; <float> [#uses=1]
  %79 = fadd float %77, %78                       ; <float> [#uses=1]
  %80 = select i1 %66, float %48, float %65       ; <float> [#uses=1]
  %81 = select i1 undef, float %80, float %79     ; <float> [#uses=1]
  %iftmp.164.0 = select i1 undef, float %29, float 1.000000e+00 ; <float> [#uses=1]
  %82 = fdiv float %81, %iftmp.164.0              ; <float> [#uses=1]
  %iftmp.165.0 = select i1 undef, float %82, float 0.000000e+00 ; <float> [#uses=1]
  store float %iftmp.165.0, float* undef, align 4
  br i1 false, label %bb4.i97, label %ccc.exit98

bb4.i97:                                          ; preds = %bb3.i
  br label %ccc.exit98

ccc.exit98:                                       ; preds = %bb4.i97, %bb3.i
  ret %bar* null

bb85:                                             ; preds = %entry
  ret %bar* null
}

declare arm_aapcs_vfpcc float @sqrtf(float) readnone
