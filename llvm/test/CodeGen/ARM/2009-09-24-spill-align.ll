; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; pr4926

define void @test_vget_lanep16() nounwind {
entry:
  %arg0_poly16x4_t = alloca <4 x i16>             ; <ptr> [#uses=1]
  %out_poly16_t = alloca i16                      ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
; CHECK: vldr
  %0 = load <4 x i16>, ptr %arg0_poly16x4_t, align 8  ; <<4 x i16>> [#uses=1]
  %1 = extractelement <4 x i16> %0, i32 1         ; <i16> [#uses=1]
  store i16 %1, ptr %out_poly16_t, align 2
  br label %return

return:                                           ; preds = %entry
  ret void
}
