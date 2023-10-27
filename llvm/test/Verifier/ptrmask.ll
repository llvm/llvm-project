; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare float @llvm.ptrmask.f32.i64(float, i64)
declare ptr @llvm.ptrmask.p0.v4i64(ptr, <4 x i64>)
declare <2 x ptr> @llvm.ptrmask.v2p0.i64(<2 x ptr>, i64)
declare <2 x ptr> @llvm.ptrmask.v2p0.v4i64(<2 x ptr>, <4 x i64>)

; CHECK: llvm.ptrmask intrinsic first argument must be pointer or vector of pointers
; CHECK-NEXT:  %1 = call float @llvm.ptrmask.f32.i64(float 0.000000e+00, i64 0)
define void @not_ptr() {
  call float @llvm.ptrmask.f32.i64(float 0.0, i64 0)
  ret void
}

; CHECK: llvm.ptrmask intrinsic arguments must be both scalars or both vectors
; CHECK: %1 = call ptr @llvm.ptrmask.p0.v4i64(ptr null, <4 x i64> zeroinitializer)
define void @scalar_vector_mismatch_1() {
  call ptr @llvm.ptrmask.p0.v4i64(ptr null, <4 x i64> zeroinitializer)
  ret void
}

; CHECK: llvm.ptrmask intrinsic arguments must be both scalars or both vectors
; CHECK: %1 = call <2 x ptr> @llvm.ptrmask.v2p0.i64(<2 x ptr> zeroinitializer, i64 0)
define void @scalar_vector_mismatch_2() {
  call <2 x ptr> @llvm.ptrmask.v2p0.i64(<2 x ptr> zeroinitializer, i64 0)
  ret void
}

; CHECK: llvm.ptrmask intrinsic arguments must have the same number of elements
; CHECK: %1 = call <2 x ptr> @llvm.ptrmask.v2p0.v4i64(<2 x ptr> zeroinitializer, <4 x i64> zeroinitializer)
define void @vector_size_mismatch() {
  call <2 x ptr> @llvm.ptrmask.v2p0.v4i64(<2 x ptr> zeroinitializer, <4 x i64> zeroinitializer)
  ret void
}
