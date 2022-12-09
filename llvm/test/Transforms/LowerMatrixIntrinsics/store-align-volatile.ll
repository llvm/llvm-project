; RUN: opt -passes='lower-matrix-intrinsics' -S < %s | FileCheck %s

define void @strided_store_volatile(<6 x i32> %in, ptr %out) {
; CHECK-LABEL: @strided_store_volatile(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> poison, <3 x i32> <i32 3, i32 4, i32 5>
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT]], ptr [[OUT:%.*]], align 4
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, ptr [[OUT]], i64 5
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT1]], ptr [[VEC_GEP]], align 4
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, ptr %out, i64 5, i1 true, i32 3, i32 2)
  ret void
}

declare void @llvm.matrix.column.major.store(<6 x i32>, ptr, i64, i1, i32, i32)


define void @multiply_store_volatile(<4 x i32> %in, ptr %out) {
; CHECK-LABEL: @multiply_store_volatile(
; CHECK:    store volatile <2 x i32> {{.*}}, ptr %out, align 4
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, ptr %out, i64 2
; CHECK-NEXT:    store volatile <2 x i32> {{.*}}, ptr [[VEC_GEP]], align 4
; CHECK-NEXT:    ret void
;
  %res = call <4 x i32> @llvm.matrix.multiply(<4 x i32> %in, <4 x i32> %in, i32 2, i32 2, i32 2)
  store volatile <4 x i32> %res, ptr %out, align 4
  ret void
}

declare <4 x i32> @llvm.matrix.multiply(<4 x i32>, <4 x i32>, i32, i32, i32)

define void @strided_store_align32(<6 x i32> %in, i64 %stride, ptr %out) {
; CHECK-LABEL: @strided_store_align32(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> poison, <3 x i32> <i32 3, i32 4, i32 5>
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, ptr [[OUT:%.*]], i64 [[VEC_START]]
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT]], ptr [[VEC_GEP]], align 32
; CHECK-NEXT:    [[VEC_START2:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP3:%.*]] = getelementptr i32, ptr [[OUT]], i64 [[VEC_START2]]
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT1]], ptr [[VEC_GEP3]], align 4
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, ptr align 32 %out, i64 %stride, i1 true, i32 3, i32 2)
  ret void
}

define void @strided_store_align2(<6 x i32> %in, i64 %stride, ptr %out) {
; CHECK-LABEL: @strided_store_align2(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> poison, <3 x i32> <i32 3, i32 4, i32 5>
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, ptr [[OUT:%.*]], i64 [[VEC_START]]
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT]], ptr [[VEC_GEP]], align 2
; CHECK-NEXT:    [[VEC_START2:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP3:%.*]] = getelementptr i32, ptr [[OUT]], i64 [[VEC_START2]]
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT1]], ptr [[VEC_GEP3]], align 2
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, ptr align 2 %out, i64 %stride, i1 true, i32 3, i32 2)
  ret void
}

define void @multiply_store_align16_stride8(<4 x i32> %in, ptr %out) {
; CHECK-LABEL: @multiply_store_align16_stride8(
; CHECK:    store <2 x i32> {{.*}}, ptr %out, align 16
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, ptr %out, i64 2
; CHECK-NEXT:    store <2 x i32> {{.*}}, ptr [[VEC_GEP]], align 8
; CHECK-NEXT:    ret void
;
  %res = call <4 x i32> @llvm.matrix.multiply(<4 x i32> %in, <4 x i32> %in, i32 2, i32 2, i32 2)
  store <4 x i32> %res, ptr %out, align 16
  ret void
}

define void @strided_store_align8_stride12(<6 x i32> %in, ptr %out) {
; CHECK-LABEL: @strided_store_align8_stride12(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> poison, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> poison, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:    [[SPLIT2:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> poison, <2 x i32> <i32 4, i32 5>
; CHECK-NEXT:    store <2 x i32> [[SPLIT]], ptr [[OUT:%.*]], align 8
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, ptr [[OUT]], i64 3
; CHECK-NEXT:    store <2 x i32> [[SPLIT1]], ptr [[VEC_GEP]], align 4
; CHECK-NEXT:    [[VEC_GEP4:%.*]] = getelementptr i32, ptr [[OUT]], i64 6
; CHECK-NEXT:    store <2 x i32> [[SPLIT2]], ptr [[VEC_GEP4]], align 8
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, ptr align 8 %out, i64 3, i1 false, i32 2, i32 3)
  ret void
}
