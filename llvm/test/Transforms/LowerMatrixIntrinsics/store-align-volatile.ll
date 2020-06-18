; RUN: opt -lower-matrix-intrinsics -S < %s | FileCheck %s
; RUN: opt -passes='lower-matrix-intrinsics' -S < %s | FileCheck %s

define void @strided_store_volatile(<6 x i32> %in, i32* %out) {
; CHECK-LABEL: @strided_store_volatile(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> undef, <3 x i32> <i32 3, i32 4, i32 5>
; CHECK-NEXT:    [[VEC_CAST:%.*]] = bitcast i32* [[OUT:%.*]] to <3 x i32>*
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT]], <3 x i32>* [[VEC_CAST]], align 4
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, i32* [[OUT]], i64 5
; CHECK-NEXT:    [[VEC_CAST2:%.*]] = bitcast i32* [[VEC_GEP]] to <3 x i32>*
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT1]], <3 x i32>* [[VEC_CAST2]], align 4
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, i32* %out, i64 5, i1 true, i32 3, i32 2)
  ret void
}

declare void @llvm.matrix.column.major.store(<6 x i32>, i32*, i64, i1, i32, i32)


define void @multiply_store_volatile(<4 x i32> %in, <4 x i32>* %out) {
; CHECK-LABEL: @multiply_store_volatile(
; CHECK:         [[TMP29:%.*]] = bitcast <4 x i32>* %out to i32*
; CHECK-NEXT:    [[VEC_CAST:%.*]] = bitcast i32* [[TMP29]] to <2 x i32>*
; CHECK-NEXT:    store volatile <2 x i32> {{.*}}, <2 x i32>* [[VEC_CAST]], align 4
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, i32* [[TMP29]], i64 2
; CHECK-NEXT:    [[VEC_CAST25:%.*]] = bitcast i32* [[VEC_GEP]] to <2 x i32>*
; CHECK-NEXT:    store volatile <2 x i32> {{.*}}, <2 x i32>* [[VEC_CAST25]], align 4
; CHECK-NEXT:    ret void
;
  %res = call <4 x i32> @llvm.matrix.multiply(<4 x i32> %in, <4 x i32> %in, i32 2, i32 2, i32 2)
  store volatile <4 x i32> %res, <4 x i32>* %out, align 4
  ret void
}

declare <4 x i32> @llvm.matrix.multiply(<4 x i32>, <4 x i32>, i32, i32, i32)

define void @strided_store_align32(<6 x i32> %in, i64 %stride, i32* %out) {
; CHECK-LABEL: @strided_store_align32(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> undef, <3 x i32> <i32 3, i32 4, i32 5>
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, i32* [[OUT:%.*]], i64 [[VEC_START]]
; CHECK-NEXT:    [[VEC_CAST:%.*]] = bitcast i32* [[VEC_GEP]] to <3 x i32>*
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT]], <3 x i32>* [[VEC_CAST]], align 4
; CHECK-NEXT:    [[VEC_START2:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP3:%.*]] = getelementptr i32, i32* [[OUT]], i64 [[VEC_START2]]
; CHECK-NEXT:    [[VEC_CAST4:%.*]] = bitcast i32* [[VEC_GEP3]] to <3 x i32>*
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT1]], <3 x i32>* [[VEC_CAST4]], align 4
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, i32* align 32 %out, i64 %stride, i1 true, i32 3, i32 2)
  ret void
}

define void @strided_store_align2(<6 x i32> %in, i64 %stride, i32* %out) {
; CHECK-LABEL: @strided_store_align2(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> undef, <3 x i32> <i32 3, i32 4, i32 5>
; CHECK-NEXT:    [[VEC_START:%.*]] = mul i64 0, [[STRIDE:%.*]]
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, i32* [[OUT:%.*]], i64 [[VEC_START]]
; CHECK-NEXT:    [[VEC_CAST:%.*]] = bitcast i32* [[VEC_GEP]] to <3 x i32>*
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT]], <3 x i32>* [[VEC_CAST]], align 4
; CHECK-NEXT:    [[VEC_START2:%.*]] = mul i64 1, [[STRIDE]]
; CHECK-NEXT:    [[VEC_GEP3:%.*]] = getelementptr i32, i32* [[OUT]], i64 [[VEC_START2]]
; CHECK-NEXT:    [[VEC_CAST4:%.*]] = bitcast i32* [[VEC_GEP3]] to <3 x i32>*
; CHECK-NEXT:    store volatile <3 x i32> [[SPLIT1]], <3 x i32>* [[VEC_CAST4]], align 4
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, i32* align 2 %out, i64 %stride, i1 true, i32 3, i32 2)
  ret void
}

define void @multiply_store_align16_stride8(<4 x i32> %in, <4 x i32>* %out) {
; CHECK-LABEL: @multiply_store_align16_stride8(
; CHECK:         [[TMP29:%.*]] = bitcast <4 x i32>* %out to i32*
; CHECK-NEXT:    [[VEC_CAST:%.*]] = bitcast i32* [[TMP29]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> {{.*}}, <2 x i32>* [[VEC_CAST]], align 4
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, i32* [[TMP29]], i64 2
; CHECK-NEXT:    [[VEC_CAST25:%.*]] = bitcast i32* [[VEC_GEP]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> {{.*}}, <2 x i32>* [[VEC_CAST25]], align 4
; CHECK-NEXT:    ret void
;
  %res = call <4 x i32> @llvm.matrix.multiply(<4 x i32> %in, <4 x i32> %in, i32 2, i32 2, i32 2)
  store <4 x i32> %res, <4 x i32>* %out, align 16
  ret void
}

define void @strided_store_align8_stride12(<6 x i32> %in, i32* %out) {
; CHECK-LABEL: @strided_store_align8_stride12(
; CHECK-NEXT:    [[SPLIT:%.*]] = shufflevector <6 x i32> [[IN:%.*]], <6 x i32> undef, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:    [[SPLIT1:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:    [[SPLIT2:%.*]] = shufflevector <6 x i32> [[IN]], <6 x i32> undef, <2 x i32> <i32 4, i32 5>
; CHECK-NEXT:    [[VEC_CAST:%.*]] = bitcast i32* [[OUT:%.*]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> [[SPLIT]], <2 x i32>* [[VEC_CAST]], align 4
; CHECK-NEXT:    [[VEC_GEP:%.*]] = getelementptr i32, i32* [[OUT]], i64 3
; CHECK-NEXT:    [[VEC_CAST3:%.*]] = bitcast i32* [[VEC_GEP]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> [[SPLIT1]], <2 x i32>* [[VEC_CAST3]], align 4
; CHECK-NEXT:    [[VEC_GEP4:%.*]] = getelementptr i32, i32* [[OUT]], i64 6
; CHECK-NEXT:    [[VEC_CAST5:%.*]] = bitcast i32* [[VEC_GEP4]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> [[SPLIT2]], <2 x i32>* [[VEC_CAST5]], align 4
; CHECK-NEXT:    ret void
;
  call void @llvm.matrix.column.major.store(<6 x i32> %in, i32* align 8 %out, i64 3, i1 false, i32 2, i32 3)
  ret void
}
