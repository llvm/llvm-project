; RUN: opt -S -dxil-op-lower %s | FileCheck %s


target triple = "dxil-pc-shadermodel6.6-compute"

define void @loadv4f32() {
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  call void @llvm.dx.updateCounter(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i8 -1)
  ret void
}
