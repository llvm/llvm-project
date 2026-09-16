; RUN: not llc -mtriple=amdgpu9.0a -filetype=null %s 2>&1 | FileCheck %s

; llvm.amdgcn.image.{sample,load,store} support 32-bit and 64-bit
; data, plus f16 under the D16 hardware conversion. Other 16-bit
; scalar types (bf16, i16) are not valid D16 data and must be
; rejected instead of silently mis-lowered or crashing.

; CHECK: error: {{.*}}unsupported image load data type
define amdgpu_ps <4 x bfloat> @sample_2d_v4bf16(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
  %v = call <4 x bfloat> @llvm.amdgcn.image.sample.2d.v4bf16.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret <4 x bfloat> %v
}

; CHECK: error: {{.*}}unsupported image load data type
define amdgpu_ps <4 x i16> @load_2d_v4i16(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
  %v = call <4 x i16> @llvm.amdgcn.image.load.2d.v4i16.i32(i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x i16> %v
}

; CHECK: error: {{.*}}unsupported image store data type
define amdgpu_ps void @store_2d_v4bf16(<8 x i32> inreg %rsrc, <4 x bfloat> %data, i32 %s, i32 %t) {
  call void @llvm.amdgcn.image.store.2d.v4bf16.i32(<4 x bfloat> %data, i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

declare <4 x bfloat> @llvm.amdgcn.image.sample.2d.v4bf16.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare <4 x i16> @llvm.amdgcn.image.load.2d.v4i16.i32(i32, i32, i32, <8 x i32>, i32, i32)
declare void @llvm.amdgcn.image.store.2d.v4bf16.i32(<4 x bfloat>, i32, i32, i32, <8 x i32>, i32, i32)
