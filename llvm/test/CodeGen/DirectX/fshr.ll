; RUN: opt -S -scalarizer -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: opt -S -scalarizer -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
;
; Make sure dxil operation function calls for funnel shifts right are generated.

; CHECK-LABEL: define{{.*}}@fshr_i16(
; CHECK-SAME: i16 %[[A:.*]], i16 %[[B:.*]], i16 %[[SHIFT:.*]])
define noundef i16 @fshr_i16(i16 %a, i16 %b, i16 %shift) {
entry:
; CHECK: %[[LEFT:.*]] = lshr i16 %[[B]], %[[SHIFT]]
; CHECK: %[[MASKED_SHIFT:.*]] = and i16 %[[SHIFT]], 15
; CHECK: %[[INVERSE_SHIFT:.*]] = sub i16 16, %[[MASKED_SHIFT]]
; CHECK: %[[RIGHT:.*]] = shl i16 %[[A]], %[[INVERSE_SHIFT]]
; CHECK: %[[RES:.*]] = or i16 %[[LEFT]], %[[RIGHT]]
; CHECK: ret i16 %[[RES]]
  %fsh = call i16 @llvm.fshr.i16(i16 %a, i16 %b, i16 %shift)
  ret i16 %fsh
}

declare i16 @llvm.fshr.i16(i16, i16, i16)

; CHECK-LABEL: define{{.*}}@fshr_v1i32(
; CHECK-SAME: <1 x i32> %[[A_VEC:.*]], <1 x i32> %[[B_VEC:.*]], <1 x i32> %[[SHIFT_VEC:.*]])
define noundef <1 x i32> @fshr_v1i32(<1 x i32> %a, <1 x i32> %b, <1 x i32> %shift) {
entry:
; CHECK: %[[A:.*]] = extractelement <1 x i32> %[[A_VEC]], i64 0
; CHECK: %[[B:.*]] = extractelement <1 x i32> %[[B_VEC]], i64 0
; CHECK: %[[SHIFT:.*]] = extractelement <1 x i32> %[[SHIFT_VEC]], i64 0
; CHECK: %[[LEFT:.*]] = lshr i32 %[[B]], %[[SHIFT]]
; CHECK: %[[MASKED_SHIFT:.*]] = and i32 %[[SHIFT]], 31
; CHECK: %[[INVERSE_SHIFT:.*]] = sub i32 32, %[[MASKED_SHIFT]]
; CHECK: %[[RIGHT:.*]] = shl i32 %[[A]], %[[INVERSE_SHIFT]]
; CHECK: %[[RES:.*]] = or i32 %[[LEFT]], %[[RIGHT]]
; CHECK: %[[RES_VEC:.*]] = insertelement <1 x i32> poison, i32 %[[RES]], i64 0
; CHECK: ret <1 x i32> %[[RES_VEC]]
  %fsh = call <1 x i32> @llvm.fshr.v1i32(<1 x i32> %a, <1 x i32> %b, <1 x i32> %shift)
  ret <1 x i32> %fsh
}

declare <1 x i32> @llvm.fshr.v1i32(<1 x i32>, <1 x i32>, <1 x i32>)

; CHECK-LABEL: define{{.*}}@fshr_v1i64(
; CHECK-SAME: <3 x i64> %[[A_VEC:.*]], <3 x i64> %[[B_VEC:.*]], <3 x i64> %[[SHIFT_VEC:.*]])
define noundef <3 x i64> @fshr_v1i64(<3 x i64> %a, <3 x i64> %b, <3 x i64> %shift) {
entry:
; CHECK: %[[A0:.*]] = extractelement <3 x i64> %[[A_VEC]], i64 0
; CHECK: %[[B0:.*]] = extractelement <3 x i64> %[[B_VEC]], i64 0
; CHECK: %[[SHIFT0:.*]] = extractelement <3 x i64> %[[SHIFT_VEC]], i64 0
; CHECK: %[[LEFT0:.*]] = lshr i64 %[[B0]], %[[SHIFT0]]
; CHECK: %[[MASKED_SHIFT0:.*]] = and i64 %[[SHIFT0]], 63
; CHECK: %[[INVERSE_SHIFT0:.*]] = sub i64 64, %[[MASKED_SHIFT0]]
; CHECK: %[[RIGHT0:.*]] = shl i64 %[[A0]], %[[INVERSE_SHIFT0]]
; CHECK: %[[RES0:.*]] = or i64 %[[LEFT0]], %[[RIGHT0]]
; 
; CHECK: %[[A1:.*]] = extractelement <3 x i64> %[[A_VEC]], i64 1
; CHECK: %[[B1:.*]] = extractelement <3 x i64> %[[B_VEC]], i64 1
; CHECK: %[[SHIFT1:.*]] = extractelement <3 x i64> %[[SHIFT_VEC]], i64 1
; CHECK: %[[LEFT1:.*]] = lshr i64 %[[B1]], %[[SHIFT1]]
; CHECK: %[[MASKED_SHIFT1:.*]] = and i64 %[[SHIFT1]], 63
; CHECK: %[[INVERSE_SHIFT1:.*]] = sub i64 64, %[[MASKED_SHIFT1]]
; CHECK: %[[RIGHT1:.*]] = shl i64 %[[A1]], %[[INVERSE_SHIFT1]]
; CHECK: %[[RES1:.*]] = or i64 %[[LEFT1]], %[[RIGHT1]]
; 
; CHECK: %[[A2:.*]] = extractelement <3 x i64> %[[A_VEC]], i64 2
; CHECK: %[[B2:.*]] = extractelement <3 x i64> %[[B_VEC]], i64 2
; CHECK: %[[SHIFT2:.*]] = extractelement <3 x i64> %[[SHIFT_VEC]], i64 2
; CHECK: %[[LEFT2:.*]] = lshr i64 %[[B2]], %[[SHIFT2]]
; CHECK: %[[MASKED_SHIFT2:.*]] = and i64 %[[SHIFT2]], 63
; CHECK: %[[INVERSE_SHIFT2:.*]] = sub i64 64, %[[MASKED_SHIFT2]]
; CHECK: %[[RIGHT2:.*]] = shl i64 %[[A2]], %[[INVERSE_SHIFT2]]
; CHECK: %[[RES2:.*]] = or i64 %[[LEFT2]], %[[RIGHT2]]
; 
; CHECK: %[[INSERT0:.*]] = insertelement <3 x i64> poison, i64 %[[RES0]], i64 0
; CHECK: %[[INSERT1:.*]] = insertelement <3 x i64> %[[INSERT0]], i64 %[[RES1]], i64 1
; CHECK: %[[RES_VEC:.*]] = insertelement <3 x i64> %[[INSERT1]], i64 %[[RES2]], i64 2
; 
; CHECK: ret <3 x i64> %[[RES_VEC]]
  %fsh = call <3 x i64> @llvm.fshr.v1i64(<3 x i64> %a, <3 x i64> %b, <3 x i64> %shift)
  ret <3 x i64> %fsh
}

declare <3 x i64> @llvm.fshr.v1i64(<3 x i64>, <3 x i64>, <3 x i64>)
