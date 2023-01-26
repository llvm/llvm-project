; RUN: llc -mtriple=armv7-eabi -mcpu=cortex-a8 < %s
; PR5411

%bar = type { %quad, float, float, [3 x ptr], [3 x ptr], [2 x ptr], [3 x i8], i8 }
%baz = type { ptr, i32 }
%foo = type { i8, %quuz, %quad, float, [64 x %quux], [128 x %bar], i32, %baz, %baz }
%quad = type { [4 x float] }
%quux = type { %quad, %quad }
%quuz = type { [4 x ptr], [4 x float], i32 }

define arm_aapcs_vfpcc ptr @aaa(ptr nocapture %this, ptr %a, ptr %b, ptr %c, i8 zeroext %forced) {
entry:
  br i1 undef, label %bb85, label %bb

bb:                                               ; preds = %entry
  %0 = getelementptr inbounds %bar, ptr null, i32 0, i32 0, i32 0, i32 2 ; <ptr> [#uses=2]
  %1 = load float, ptr undef, align 4                 ; <float> [#uses=1]
  %2 = fsub float 0.000000e+00, undef             ; <float> [#uses=2]
  %3 = fmul float 0.000000e+00, undef             ; <float> [#uses=1]
  %4 = load float, ptr %0, align 4                    ; <float> [#uses=3]
  %5 = fmul float %4, %2                          ; <float> [#uses=1]
  %6 = fsub float %3, %5                          ; <float> [#uses=1]
  %7 = fmul float %4, undef                       ; <float> [#uses=1]
  %8 = fsub float %7, undef                       ; <float> [#uses=1]
  %9 = fmul float undef, %2                       ; <float> [#uses=1]
  %10 = fmul float 0.000000e+00, undef            ; <float> [#uses=1]
  %11 = fsub float %9, %10                        ; <float> [#uses=1]
  %12 = fmul float undef, %6                      ; <float> [#uses=1]
  %13 = fmul float 0.000000e+00, %8               ; <float> [#uses=1]
  %14 = fadd float %12, %13                       ; <float> [#uses=1]
  %15 = fmul float %1, %11                        ; <float> [#uses=1]
  %16 = fadd float %14, %15                       ; <float> [#uses=1]
  %17 = select i1 undef, float undef, float %16   ; <float> [#uses=1]
  %18 = fdiv float %17, 0.000000e+00              ; <float> [#uses=1]
  store float %18, ptr undef, align 4
  %19 = fmul float %4, undef                      ; <float> [#uses=1]
  store float %19, ptr %0, align 4
  ret ptr null

bb85:                                             ; preds = %entry
  ret ptr null
}
