; RUN: opt -passes='lower-matrix-intrinsics' -S < %s | FileCheck %s

define <9 x double> @strided_load_3x3_volatile(ptr %in, i64 %stride) {
; CHECK-LABEL: @strided_load_3x3_volatile(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START]]
; CHECK-NEXT:    load volatile <3 x double>, ptr [[VEC_GEP]], align 8
; CHECK-NEXT:    [[VEC_START1:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP2:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START1]]
; CHECK-NEXT:    load volatile <3 x double>, ptr [[VEC_GEP2]], align 8
; CHECK-NEXT:    [[VEC_START5:%.*]] = mul i64 2, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP6:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START5]]
; CHECK-NEXT:    load volatile <3 x double>, ptr [[VEC_GEP6]], align 8
; CHECK-NOT:     = load
;
entry:
  %load = call <9 x double> @llvm.matrix.column.major.load.v9f64(ptr %in, i64 %stride, i1 true, i32 3, i32 3)
  ret <9 x double> %load
}

declare <9 x double> @llvm.matrix.column.major.load.v9f64(ptr, i64, i1, i32, i32)

define <4 x double> @load_volatile_multiply(ptr %in) {
; CHECK-LABEL: @load_volatile_multiply(
; CHECK-NEXT:    load volatile <2 x double>, ptr [[IN:%.*]], align 8
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr double, ptr [[IN]], i64 2
; CHECK-NEXT:    load volatile <2 x double>, ptr [[VEC_GEP]], align 8
; CHECK-NOT:     = load
;
  %in.m = load volatile <4 x double>, ptr %in, align 8
  %res = call <4 x double> @llvm.matrix.multiply(<4 x double> %in.m, <4 x double> %in.m, i32 2, i32 2, i32 2)
  ret <4 x double> %res
}

declare <4 x double> @llvm.matrix.multiply(<4 x double>, <4 x double>, i32, i32, i32)


define <9 x double> @strided_load_3x3_align32(ptr %in, i64 %stride) {
; CHECK-LABEL: @strided_load_3x3_align32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START]]
; CHECK-NEXT:    load <3 x double>, ptr [[VEC_GEP]], align 32
; CHECK-NEXT:    [[VEC_START1:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP2:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START1]]
; CHECK-NEXT:    load <3 x double>, ptr [[VEC_GEP2]], align 8
; CHECK-NEXT:    [[VEC_START5:%.*]] = mul i64 2, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP6:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START5]]
; CHECK-NEXT:    load <3 x double>, ptr [[VEC_GEP6]], align 8
; CHECK-NOT:     = load
;
entry:
  %load = call <9 x double> @llvm.matrix.column.major.load.v9f64(ptr align 32 %in, i64 %stride, i1 false, i32 3, i32 3)
  ret <9 x double> %load
}

define <9 x double> @strided_load_3x3_align2(ptr %in, i64 %stride) {
; CHECK-LABEL: @strided_load_3x3_align2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START]]
; CHECK-NEXT:    load <3 x double>, ptr [[VEC_GEP]], align 2
; CHECK-NEXT:    [[VEC_START1:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP2:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START1]]
; CHECK-NEXT:    load <3 x double>, ptr [[VEC_GEP2]], align 2
; CHECK-NEXT:    [[VEC_START5:%.*]] = mul i64 2, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP6:%.*]] = getelementptr double, ptr %in, i64 [[VEC_START5]]
; CHECK-NEXT:    load <3 x double>, ptr [[VEC_GEP6]], align 2
; CHECK-NOT:     = load
;
entry:
  %load = call <9 x double> @llvm.matrix.column.major.load.v9f64(ptr align 2 %in, i64 %stride, i1 false, i32 3, i32 3)
  ret <9 x double> %load
}


define <4 x double> @load_align2_multiply(ptr %in) {
; CHECK-LABEL: @load_align2_multiply(
; CHECK-NEXT:    load <2 x double>, ptr [[IN:%.*]], align 2
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr double, ptr [[IN]], i64 2
; CHECK-NEXT:    load <2 x double>, ptr [[VEC_GEP]], align 2
; CHECK-NOT:     = load
;
  %in.m = load <4 x double>, ptr %in, align 2
  %res = call <4 x double> @llvm.matrix.multiply(<4 x double> %in.m, <4 x double> %in.m, i32 2, i32 2, i32 2)
  ret <4 x double> %res
}

define <6 x float> @strided_load_2x3_align16_stride2(ptr %in) {
; CHECK-LABEL: @strided_load_2x3_align16_stride2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[COL_LOAD:%.*]] = load <2 x float>, ptr %in, align 16
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr float, ptr %in, i64 2
; CHECK-NEXT:    [[COL_LOAD2:%.*]] = load <2 x float>, ptr [[VEC_GEP]], align 8
; CHECK-NEXT:    [[VEC_GEP3:%.*]] = getelementptr float, ptr %in, i64 4
; CHECK-NEXT:    [[COL_LOAD5:%.*]] = load <2 x float>, ptr [[VEC_GEP3]], align 16
; CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <2 x float> [[COL_LOAD]], <2 x float> [[COL_LOAD2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <2 x float> [[COL_LOAD5]], <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <4 x float> [[TMP1]], <4 x float> [[TMP2]], <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
; CHECK-NEXT:    ret <6 x float> [[TMP3]]
;
entry:
  %load = call <6 x float> @llvm.matrix.column.major.load.v6f32(ptr align 16 %in, i64 2, i1 false, i32 2, i32 3)
  ret <6 x float> %load
}

declare <6 x float> @llvm.matrix.column.major.load.v6f32(ptr, i64, i1, i32, i32)
