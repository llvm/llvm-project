; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
;
; Make sure dxil operation function calls for funnel shifts left are generated.

; CHECK-LABEL: define{{.*}}@fshl_i16(
; CHECK-SAME: i16 %[[A:.*]], i16 %[[B:.*]], i16 %[[SHIFT:.*]])
define noundef i16 @fshl_i16(i16 %a, i16 %b, i16 %shift) {
entry:
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[MASKED_SHIFT:.*]] = and i16 %[[SHIFT]], 15
; CHECK-NEXT:   %[[NOT_SHIFT:.*]] = xor i16 %[[SHIFT]], -1
; CHECK-NEXT:   %[[INVERSE_SHIFT:.*]] = and i16 %[[NOT_SHIFT]], 15
; CHECK-NEXT:   %[[LEFT:.*]] = shl i16 %[[A]], %[[MASKED_SHIFT]]
; CHECK-NEXT:   %[[SHIFT_B_1:.*]] = lshr i16 %[[B]], 1
; CHECK-NEXT:   %[[RIGHT:.*]] = lshr i16 %[[SHIFT_B_1]], %[[INVERSE_SHIFT]]
; CHECK-NEXT:   %[[RES:.*]] = or i16 %[[LEFT]], %[[RIGHT]]
; CHECK-NEXT:   ret i16 %[[RES]]
  %fsh = call i16 @llvm.fshl.i16(i16 %a, i16 %b, i16 %shift)
  ret i16 %fsh
}

declare i16 @llvm.fshl.i16(i16, i16, i16)

; CHECK-LABEL: define{{.*}}@fshl_v1i32(
; CHECK-SAME: <1 x i32> %[[A_VEC:.*]], <1 x i32> %[[B_VEC:.*]], <1 x i32> %[[SHIFT_VEC:.*]])
define noundef <1 x i32> @fshl_v1i32(<1 x i32> %a, <1 x i32> %b, <1 x i32> %shift) {
entry:
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[B:.*]] = extractelement <1 x i32> %[[B_VEC]], i64 0
; CHECK-NEXT:   %[[A:.*]] = extractelement <1 x i32> %[[A_VEC]], i64 0
; CHECK-NEXT:   %[[SHIFT:.*]] = extractelement <1 x i32> %[[SHIFT_VEC]], i64 0
; CHECK-NEXT:   %[[MASKED_SHIFT:.*]] = and i32 %[[SHIFT]], 31
; CHECK-NEXT:   %[[NOT_SHIFT:.*]] = xor i32 %[[SHIFT]], -1
; CHECK-NEXT:   %[[INVERSE_SHIFT:.*]] = and i32 %[[NOT_SHIFT]], 31
; CHECK-NEXT:   %[[LEFT:.*]] = shl i32 %[[A]], %[[MASKED_SHIFT]]
; CHECK-NEXT:   %[[SHIFT_B_1:.*]] = lshr i32 %[[B]], 1
; CHECK-NEXT:   %[[RIGHT:.*]] = lshr i32 %[[SHIFT_B_1]], %[[INVERSE_SHIFT]]
; CHECK-NEXT:   %[[RES:.*]] = or i32 %[[LEFT]], %[[RIGHT]]
; CHECK-NEXT:   %[[RES_VEC:.*]] = insertelement <1 x i32> poison, i32 %[[RES]], i64 0
; CHECK-NEXT:   ret <1 x i32> %[[RES_VEC]]
  %fsh = call <1 x i32> @llvm.fshl.v1i32(<1 x i32> %a, <1 x i32> %b, <1 x i32> %shift)
  ret <1 x i32> %fsh
}

declare <1 x i32> @llvm.fshl.v1i32(<1 x i32>, <1 x i32>, <1 x i32>)

; CHECK-LABEL: define{{.*}}@fshl_v1i64(
; CHECK-SAME: <3 x i64> %[[A_VEC:.*]], <3 x i64> %[[B_VEC:.*]], <3 x i64> %[[SHIFT_VEC:.*]])
define noundef <3 x i64> @fshl_v1i64(<3 x i64> %a, <3 x i64> %b, <3 x i64> %shift) {
entry:
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[B0:.*]] = extractelement <3 x i64> %[[B_VEC]], i64 0
; CHECK-NEXT:   %[[B1:.*]] = extractelement <3 x i64> %[[B_VEC]], i64 1
; CHECK-NEXT:   %[[B2:.*]] = extractelement <3 x i64> %[[B_VEC]], i64 2
;
; CHECK-NEXT:   %[[A0:.*]] = extractelement <3 x i64> %[[A_VEC]], i64 0
; CHECK-NEXT:   %[[A1:.*]] = extractelement <3 x i64> %[[A_VEC]], i64 1
; CHECK-NEXT:   %[[A2:.*]] = extractelement <3 x i64> %[[A_VEC]], i64 2
; 
; CHECK-NEXT:   %[[SHIFT0:.*]] = extractelement <3 x i64> %[[SHIFT_VEC]], i64 0
; CHECK-NEXT:   %[[MASKED_SHIFT0:.*]] = and i64 %[[SHIFT0]], 63
; CHECK-NEXT:   %[[SHIFT1:.*]] = extractelement <3 x i64> %[[SHIFT_VEC]], i64 1
; CHECK-NEXT:   %[[MASKED_SHIFT1:.*]] = and i64 %[[SHIFT1]], 63
; CHECK-NEXT:   %[[SHIFT2:.*]] = extractelement <3 x i64> %[[SHIFT_VEC]], i64 2
; CHECK-NEXT:   %[[MASKED_SHIFT2:.*]] = and i64 %[[SHIFT2]], 63
; 
; CHECK-NEXT:   %[[NOT_SHIFT0:.*]] = xor i64 %[[SHIFT0]], -1
; CHECK-NEXT:   %[[NOT_SHIFT1:.*]] = xor i64 %[[SHIFT1]], -1
; CHECK-NEXT:   %[[NOT_SHIFT2:.*]] = xor i64 %[[SHIFT2]], -1
; 
; CHECK-NEXT:   %[[INVERSE_SHIFT0:.*]] = and i64 %[[NOT_SHIFT0]], 63
; CHECK-NEXT:   %[[INVERSE_SHIFT1:.*]] = and i64 %[[NOT_SHIFT1]], 63
; CHECK-NEXT:   %[[INVERSE_SHIFT2:.*]] = and i64 %[[NOT_SHIFT2]], 63
;
; CHECK-NEXT:   %[[LEFT0:.*]] = shl i64 %[[A0]], %[[MASKED_SHIFT0]]
; CHECK-NEXT:   %[[LEFT1:.*]] = shl i64 %[[A1]], %[[MASKED_SHIFT1]]
; CHECK-NEXT:   %[[LEFT2:.*]] = shl i64 %[[A2]], %[[MASKED_SHIFT2]]
;
; CHECK-NEXT:   %[[SHIFT_B0_1:.*]] = lshr i64 %[[B0]], 1
; CHECK-NEXT:   %[[SHIFT_B1_1:.*]] = lshr i64 %[[B1]], 1
; CHECK-NEXT:   %[[SHIFT_B2_1:.*]] = lshr i64 %[[B2]], 1
; 
; CHECK-NEXT:   %[[RIGHT0:.*]] = lshr i64 %[[SHIFT_B0_1]], %[[INVERSE_SHIFT0]]
; CHECK-NEXT:   %[[RIGHT1:.*]] = lshr i64 %[[SHIFT_B1_1]], %[[INVERSE_SHIFT1]]
; CHECK-NEXT:   %[[RIGHT2:.*]] = lshr i64 %[[SHIFT_B2_1]], %[[INVERSE_SHIFT2]]
;
; CHECK-NEXT:   %[[RES0:.*]] = or i64 %[[LEFT0]], %[[RIGHT0]]
; CHECK-NEXT:   %[[RES1:.*]] = or i64 %[[LEFT1]], %[[RIGHT1]]
; CHECK-NEXT:   %[[RES2:.*]] = or i64 %[[LEFT2]], %[[RIGHT2]]
; 
; CHECK-NEXT:   %[[INSERT0:.*]] = insertelement <3 x i64> poison, i64 %[[RES0]], i64 0
; CHECK-NEXT:   %[[INSERT1:.*]] = insertelement <3 x i64> %[[INSERT0]], i64 %[[RES1]], i64 1
; CHECK-NEXT:   %[[RES_VEC:.*]] = insertelement <3 x i64> %[[INSERT1]], i64 %[[RES2]], i64 2
; 
; CHECK-NEXT: ret <3 x i64> %[[RES_VEC]]
  %fsh = call <3 x i64> @llvm.fshl.v1i64(<3 x i64> %a, <3 x i64> %b, <3 x i64> %shift)
  ret <3 x i64> %fsh
}

declare <3 x i64> @llvm.fshl.v1i64(<3 x i64>, <3 x i64>, <3 x i64>)
