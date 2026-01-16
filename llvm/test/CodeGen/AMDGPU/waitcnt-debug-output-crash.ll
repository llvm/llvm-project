; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -debug-only si-insert-waitcnts < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: Begin Block: bb.0.bb

define amdgpu_kernel void @main(ptr addrspace(3) %arg) {
bb:
  %i = load <16 x i8>, ptr addrspace(3) %arg, align 16
  tail call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %i1 = shufflevector <16 x i8> %i, <16 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i2 = shufflevector <64 x i8> zeroinitializer, <64 x i8> %i1, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>
  fence syncscope("workgroup") release
  %i3 = bitcast <64 x i8> %i2 to <16 x i32>
  %i4 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %i3, i32 0, <16 x i32> zeroinitializer, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i5 = extractelement <8 x float> %i4, i64 0
  %i6 = insertelement <4 x float> zeroinitializer, float %i5, i64 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %i6, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  ret void
}
