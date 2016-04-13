; RUN: opt < %s -instcombine -S | FileCheck %s

;; MASKED LOADS

; If the mask isn't constant, do nothing.

define <4 x float> @mload(i8* %f, <4 x i32> %mask) {
  %ld = tail call <4 x float> @llvm.x86.avx.maskload.ps(i8* %f, <4 x i32> %mask)
  ret <4 x float> %ld

; CHECK-LABEL: @mload(
; CHECK-NEXT:  %ld = tail call <4 x float> @llvm.x86.avx.maskload.ps(i8* %f, <4 x i32> %mask)
; CHECK-NEXT:  ret <4 x float> %ld
}

; Zero mask returns a zero vector.

define <4 x float> @mload_zeros(i8* %f) {
  %ld = tail call <4 x float> @llvm.x86.avx.maskload.ps(i8* %f, <4 x i32> zeroinitializer)
  ret <4 x float> %ld

; CHECK-LABEL: @mload_zeros(
; CHECK-NEXT:  ret <4 x float> zeroinitializer
}

; Only the sign bit matters.

define <4 x float> @mload_fake_ones(i8* %f) {
  %ld = tail call <4 x float> @llvm.x86.avx.maskload.ps(i8* %f, <4 x i32> <i32 1, i32 2, i32 3, i32 2147483647>)
  ret <4 x float> %ld

; CHECK-LABEL: @mload_fake_ones(
; CHECK-NEXT:  ret <4 x float> zeroinitializer
}

; All mask bits are set, so this is just a vector load.

define <4 x float> @mload_real_ones(i8* %f) {
  %ld = tail call <4 x float> @llvm.x86.avx.maskload.ps(i8* %f, <4 x i32> <i32 -1, i32 -2, i32 -3, i32 2147483648>)
  ret <4 x float> %ld

; CHECK-LABEL: @mload_real_ones(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x float>*
; CHECK-NEXT:  %unmaskedload = load <4 x float>, <4 x float>* %castvec
; CHECK-NEXT:  ret <4 x float> %unmaskedload
}

; It's a constant mask, so convert to an LLVM intrinsic. The backend should optimize further.

define <4 x float> @mload_one_one(i8* %f) {
  %ld = tail call <4 x float> @llvm.x86.avx.maskload.ps(i8* %f, <4 x i32> <i32 0, i32 0, i32 0, i32 -1>)
  ret <4 x float> %ld

; CHECK-LABEL: @mload_one_one(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x float>*
; CHECK-NEXT:  %1 = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %castvec, i32 1, <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x float> zeroinitializer)
; CHECK-NEXT:  ret <4 x float> %1
}

; Try doubles.

define <2 x double> @mload_one_one_double(i8* %f) {
  %ld = tail call <2 x double> @llvm.x86.avx.maskload.pd(i8* %f, <2 x i64> <i64 -1, i64 0>)
  ret <2 x double> %ld

; CHECK-LABEL: @mload_one_one_double(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <2 x double>*
; CHECK-NEXT:  %1 = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %castvec, i32 1, <2 x i1> <i1 true, i1 false>, <2 x double> zeroinitializer)
; CHECK-NEXT:  ret <2 x double> %1
}

; Try 256-bit FP ops.

define <8 x float> @mload_v8f32(i8* %f) {
  %ld = tail call <8 x float> @llvm.x86.avx.maskload.ps.256(i8* %f, <8 x i32> <i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 0>)
  ret <8 x float> %ld

; CHECK-LABEL: @mload_v8f32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <8 x float>*
; CHECK-NEXT:  %1 = call <8 x float> @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %castvec, i32 1, <8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false>, <8 x float> zeroinitializer)
; CHECK-NEXT:  ret <8 x float> %1
}

define <4 x double> @mload_v4f64(i8* %f) {
  %ld = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %f, <4 x i64> <i64 -1, i64 0, i64 0, i64 0>)
  ret <4 x double> %ld

; CHECK-LABEL: @mload_v4f64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x double>*
; CHECK-NEXT:  %1 = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %castvec, i32 1, <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x double> zeroinitializer)
; CHECK-NEXT:  ret <4 x double> %1
}

; Try the AVX2 variants.

define <4 x i32> @mload_v4i32(i8* %f) {
  %ld = tail call <4 x i32> @llvm.x86.avx2.maskload.d(i8* %f, <4 x i32> <i32 0, i32 0, i32 0, i32 -1>)
  ret <4 x i32> %ld

; CHECK-LABEL: @mload_v4i32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x i32>*
; CHECK-NEXT:  %1 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %castvec, i32 1, <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32> zeroinitializer)
; CHECK-NEXT:  ret <4 x i32> %1
}

define <2 x i64> @mload_v2i64(i8* %f) {
  %ld = tail call <2 x i64> @llvm.x86.avx2.maskload.q(i8* %f, <2 x i64> <i64 -1, i64 0>)
  ret <2 x i64> %ld

; CHECK-LABEL: @mload_v2i64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <2 x i64>*
; CHECK-NEXT:  %1 = call <2 x i64> @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %castvec, i32 1, <2 x i1> <i1 true, i1 false>, <2 x i64> zeroinitializer)
; CHECK-NEXT:  ret <2 x i64> %1
}

define <8 x i32> @mload_v8i32(i8* %f) {
  %ld = tail call <8 x i32> @llvm.x86.avx2.maskload.d.256(i8* %f, <8 x i32> <i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 0>)
  ret <8 x i32> %ld

; CHECK-LABEL: @mload_v8i32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <8 x i32>*
; CHECK-NEXT:  %1 = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %castvec, i32 1, <8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false>, <8 x i32> zeroinitializer)
; CHECK-NEXT:  ret <8 x i32> %1
}

define <4 x i64> @mload_v4i64(i8* %f) {
  %ld = tail call <4 x i64> @llvm.x86.avx2.maskload.q.256(i8* %f, <4 x i64> <i64 -1, i64 0, i64 0, i64 0>)
  ret <4 x i64> %ld

; CHECK-LABEL: @mload_v4i64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x i64>*
; CHECK-NEXT:  %1 = call <4 x i64> @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %castvec, i32 1, <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i64> zeroinitializer)
; CHECK-NEXT:  ret <4 x i64> %1
}


;; MASKED STORES

; If the mask isn't constant, do nothing.

define void @mstore(i8* %f, <4 x i32> %mask, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> %mask, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore(
; CHECK-NEXT:  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> %mask, <4 x float> %v)
; CHECK-NEXT:  ret void
}

; Zero mask is a nop.

define void @mstore_zeros(i8* %f, <4 x float> %v)  {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> zeroinitializer, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_zeros(
; CHECK-NEXT:  ret void
}

; Only the sign bit matters.

define void @mstore_fake_ones(i8* %f, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> <i32 1, i32 2, i32 3, i32 2147483647>, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_fake_ones(
; CHECK-NEXT:  ret void
}

; All mask bits are set, so this is just a vector store.

define void @mstore_real_ones(i8* %f, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> <i32 -1, i32 -2, i32 -3, i32 -2147483648>, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_real_ones(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x float>*
; CHECK-NEXT:  store <4 x float> %v, <4 x float>* %castvec
; CHECK-NEXT:  ret void
}

; It's a constant mask, so convert to an LLVM intrinsic. The backend should optimize further.

define void @mstore_one_one(i8* %f, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> <i32 0, i32 0, i32 0, i32 -1>, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_one_one(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x float>*
; CHECK-NEXT:  call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %v, <4 x float>* %castvec, i32 1, <4 x i1> <i1 false, i1 false, i1 false, i1 true>)
; CHECK-NEXT:  ret void
}

; Try doubles.

define void @mstore_one_one_double(i8* %f, <2 x double> %v) {
  tail call void @llvm.x86.avx.maskstore.pd(i8* %f, <2 x i64> <i64 -1, i64 0>, <2 x double> %v)
  ret void

; CHECK-LABEL: @mstore_one_one_double(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <2 x double>*
; CHECK-NEXT:  call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %v, <2 x double>* %castvec, i32 1, <2 x i1> <i1 true, i1 false>)
; CHECK-NEXT:  ret void
}

; Try 256-bit FP ops.

define void @mstore_v8f32(i8* %f, <8 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps.256(i8* %f, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 -1, i32 -2, i32 -3, i32 -4>, <8 x float> %v)
  ret void

; CHECK-LABEL: @mstore_v8f32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <8 x float>*
; CHECK-NEXT:  call void @llvm.masked.store.v8f32.p0v8f32(<8 x float> %v, <8 x float>* %castvec, i32 1, <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:  ret void
}

define void @mstore_v4f64(i8* %f, <4 x double> %v) {
  tail call void @llvm.x86.avx.maskstore.pd.256(i8* %f, <4 x i64> <i64 -1, i64 0, i64 1, i64 2>, <4 x double> %v)
  ret void

; CHECK-LABEL: @mstore_v4f64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x double>*
; CHECK-NEXT:  call void @llvm.masked.store.v4f64.p0v4f64(<4 x double> %v, <4 x double>* %castvec, i32 1, <4 x i1> <i1 true, i1 false, i1 false, i1 false>)
; CHECK-NEXT:  ret void
}

; Try the AVX2 variants.

define void @mstore_v4i32(i8* %f, <4 x i32> %v) {
  tail call void @llvm.x86.avx2.maskstore.d(i8* %f, <4 x i32> <i32 0, i32 1, i32 -1, i32 -2>, <4 x i32> %v)
  ret void

; CHECK-LABEL: @mstore_v4i32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x i32>*
; CHECK-NEXT:  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %v, <4 x i32>* %castvec, i32 1, <4 x i1> <i1 false, i1 false, i1 true, i1 true>)
; CHECK-NEXT:  ret void
}

define void @mstore_v2i64(i8* %f, <2 x i64> %v) {
  tail call void @llvm.x86.avx2.maskstore.q(i8* %f, <2 x i64> <i64 -1, i64 0>, <2 x i64> %v)
  ret void

; CHECK-LABEL: @mstore_v2i64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <2 x i64>*
; CHECK-NEXT:  call void @llvm.masked.store.v2i64.p0v2i64(<2 x i64> %v, <2 x i64>* %castvec, i32 1, <2 x i1> <i1 true, i1 false>)
; CHECK-NEXT:  ret void
}

define void @mstore_v8i32(i8* %f, <8 x i32> %v) {
  tail call void @llvm.x86.avx2.maskstore.d.256(i8* %f, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 -1, i32 -2, i32 -3, i32 -4>, <8 x i32> %v)
  ret void

; CHECK-LABEL: @mstore_v8i32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <8 x i32>*
; CHECK-NEXT:  call void @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %v, <8 x i32>* %castvec, i32 1, <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:  ret void
}

define void @mstore_v4i64(i8* %f, <4 x i64> %v) {
  tail call void @llvm.x86.avx2.maskstore.q.256(i8* %f, <4 x i64> <i64 -1, i64 0, i64 1, i64 2>, <4 x i64> %v)
  ret void

; CHECK-LABEL: @mstore_v4i64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x i64>*
; CHECK-NEXT:  call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %v, <4 x i64>* %castvec, i32 1, <4 x i1> <i1 true, i1 false, i1 false, i1 false>)
; CHECK-NEXT:  ret void
}

; The original SSE2 masked store variant.

define void @mstore_v16i8_sse2_zeros(<16 x i8> %d, i8* %p) {
  tail call void @llvm.x86.sse2.maskmov.dqu(<16 x i8> %d, <16 x i8> zeroinitializer, i8* %p)
  ret void

; CHECK-LABEL: @mstore_v16i8_sse2_zeros(
; CHECK-NEXT:  ret void
}


declare <4 x float> @llvm.x86.avx.maskload.ps(i8*, <4 x i32>)
declare <2 x double> @llvm.x86.avx.maskload.pd(i8*, <2 x i64>)
declare <8 x float> @llvm.x86.avx.maskload.ps.256(i8*, <8 x i32>)
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8*, <4 x i64>)

declare <4 x i32> @llvm.x86.avx2.maskload.d(i8*, <4 x i32>)
declare <2 x i64> @llvm.x86.avx2.maskload.q(i8*, <2 x i64>)
declare <8 x i32> @llvm.x86.avx2.maskload.d.256(i8*, <8 x i32>)
declare <4 x i64> @llvm.x86.avx2.maskload.q.256(i8*, <4 x i64>)

declare void @llvm.x86.avx.maskstore.ps(i8*, <4 x i32>, <4 x float>)
declare void @llvm.x86.avx.maskstore.pd(i8*, <2 x i64>, <2 x double>)
declare void @llvm.x86.avx.maskstore.ps.256(i8*, <8 x i32>, <8 x float>)
declare void @llvm.x86.avx.maskstore.pd.256(i8*, <4 x i64>, <4 x double>)

declare void @llvm.x86.avx2.maskstore.d(i8*, <4 x i32>, <4 x i32>)
declare void @llvm.x86.avx2.maskstore.q(i8*, <2 x i64>, <2 x i64>)
declare void @llvm.x86.avx2.maskstore.d.256(i8*, <8 x i32>, <8 x i32>)
declare void @llvm.x86.avx2.maskstore.q.256(i8*, <4 x i64>, <4 x i64>)

declare void @llvm.x86.sse2.maskmov.dqu(<16 x i8>, <16 x i8>, i8*)

