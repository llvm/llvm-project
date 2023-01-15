; RUN: opt -S -passes=amdgpu-image-intrinsic-opt -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1010 < %s | FileCheck -check-prefixes=NO-MSAA %s
; RUN: opt -S -passes=amdgpu-image-intrinsic-opt -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1100 < %s | FileCheck -check-prefixes=NO-MSAA %s
; RUN: opt -S -passes=amdgpu-image-intrinsic-opt -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1150 < %s | FileCheck -check-prefixes=MSAA %s

; NO-MSAA-NOT:  @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32
; NO-MSAA-NOT:  @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i32

; MSAA-LABEL: @load_2dmsaa_v4f32_dmask1
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_dmask1(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4f32_dmask2
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_dmask2(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 2, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 2, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 2, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 2, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4f32_dmask4
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_dmask4(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 4, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 4, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 4, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 4, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4f32_dmask8
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 8, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_dmask8(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 8, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 8, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 8, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 8, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4f32_reverse
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 3
; MSAA:  %2 = extractelement <4 x float> %0, i64 2
; MSAA:  %3 = extractelement <4 x float> %0, i64 1
; MSAA:  %4 = extractelement <4 x float> %0, i64 0
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_reverse(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; Don't combine because the vaddr inputs are not identical.
; MSAA-LABEL: @load_2dmsaa_v4f32_vaddr
; MSAA-NOT:  @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_vaddr(<8 x i32> inreg %rsrc, i32 %s0, i32 %t0, i32 %s1, i32 %t1, i32 %s2, i32 %t2, i32 %s3, i32 %t3) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s0, i32 %t0, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s1, i32 %t1, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s2, i32 %t2, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s3, i32 %t3, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v8f32
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [8 x float] @load_2dmsaa_v8f32(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i5 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i6 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i7 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i8 = insertvalue [8 x float] undef, float %i, 0
  %i9 = insertvalue [8 x float] %i8, float %i1, 1
  %i10 = insertvalue [8 x float] %i9, float %i2, 2
  %i11 = insertvalue [8 x float] %i10, float %i3, 3
  %i12 = insertvalue [8 x float] %i11, float %i4, 4
  %i13 = insertvalue [8 x float] %i12, float %i5, 5
  %i14 = insertvalue [8 x float] %i13, float %i6, 6
  %i15 = insertvalue [8 x float] %i14, float %i7, 7
  ret [8 x float] %i15
}

; MSAA-LABEL: @load_2dmsaa_v4f32_interleaved
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_interleaved(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = insertvalue [4 x float] undef, float %i, 0
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = insertvalue [4 x float] %i1, float %i2, 1
  %i4 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i5 = insertvalue [4 x float] %i3, float %i4, 2
  %i6 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i7 = insertvalue [4 x float] %i5, float %i6, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v2f32_fragId01
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
define amdgpu_ps [2 x float] @load_2dmsaa_v2f32_fragId01(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = insertvalue [2 x float] undef, float %i, 0
  %i3 = insertvalue [2 x float] %i2, float %i1, 1
  ret [2 x float] %i3
}

; MSAA-LABEL: @load_2dmsaa_v2f32_fragId23
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 2
; MSAA:  %2 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [2 x float] @load_2dmsaa_v2f32_fragId23(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = insertvalue [2 x float] undef, float %i, 0
  %i3 = insertvalue [2 x float] %i2, float %i1, 1
  ret [2 x float] %i3
}

; Don't combine because it's not profitable: the resulting msaa loads would
; have 8 vdata outputs.
; MSAA-LABEL: @load_2dmsaa_v2v2f32_dmask3
; MSAA-NOT:  @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32
define amdgpu_ps [2 x <2 x float>] @load_2dmsaa_v2v2f32_dmask3(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [2 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [2 x <2 x float>] %i4, <2 x float> %i1, 1
  ret [2 x <2 x float>] %i5
}

; MSAA-LABEL: @load_2dmsaa_v4v2f32_dmask3

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2dmsaa_v4v2f32_dmask3(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v2f32_dmask5

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2dmsaa_v4v2f32_dmask5(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 5, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 5, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 5, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 5, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v2f32_dmask6

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2dmsaa_v4v2f32_dmask6(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 6, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 6, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 6, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 6, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v2f32_dmask9

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 8, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2dmsaa_v4v2f32_dmask9(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 9, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 9, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 9, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 9, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v2f32_dmask10

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 8, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2dmsaa_v4v2f32_dmask10(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 10, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 10, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 10, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 10, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v2f32_dmask12

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 8, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2dmsaa_v4v2f32_dmask12(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 12, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 12, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 12, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32 12, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v2f16_fragId01
; MSAA:  %0 = call <4 x half> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f16.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x half> %0, i64 0
; MSAA:  %2 = extractelement <4 x half> %0, i64 1
define amdgpu_ps [2 x half] @load_2dmsaa_v2f16_fragId01(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call half @llvm.amdgcn.image.load.2dmsaa.f16.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call half @llvm.amdgcn.image.load.2dmsaa.f16.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = insertvalue [2 x half] undef, half %i, 0
  %i3 = insertvalue [2 x half] %i2, half %i1, 1
  ret [2 x half] %i3
}

; MSAA-LABEL: @load_2darraymsaa_v4f32_dmask1
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
define amdgpu_ps [4 x float] @load_2darraymsaa_v4f32_dmask1(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2darraymsaa_v4v2f32_dmask3

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 %slice, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %2 = extractelement <4 x float> %0, i64 0
; MSAA:  %3 = insertelement <2 x float> undef, float %2, i64 0
; MSAA:  %4 = extractelement <4 x float> %1, i64 0
; MSAA:  %5 = insertelement <2 x float> %3, float %4, i64 1

; MSAA:  %6 = extractelement <4 x float> %0, i64 1
; MSAA:  %7 = insertelement <2 x float> undef, float  %6, i64 0
; MSAA:  %8 = extractelement <4 x float> %1, i64 1
; MSAA:  %9 = insertelement <2 x float> %7, float %8, i64 1
define amdgpu_ps [4 x <2 x float>] @load_2darraymsaa_v4v2f32_dmask3(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %i = call <2 x float> @llvm.amdgcn.image.load.2darraymsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 %slice, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <2 x float> @llvm.amdgcn.image.load.2darraymsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 %slice, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <2 x float> @llvm.amdgcn.image.load.2darraymsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 %slice, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <2 x float> @llvm.amdgcn.image.load.2darraymsaa.v2f32.i32(i32 3, i32 %s, i32 %t, i32 %slice, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <2 x float>] undef, <2 x float> %i, 0
  %i5 = insertvalue [4 x <2 x float>] %i4, <2 x float> %i1, 1
  %i6 = insertvalue [4 x <2 x float>] %i5, <2 x float> %i2, 2
  %i7 = insertvalue [4 x <2 x float>] %i6, <2 x float> %i3, 3
  ret [4 x <2 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v3f32_dmask7

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %2 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %3 = extractelement <4 x float> %0, i64 0
; MSAA:  %4 = insertelement <3 x float> undef, float %3, i64 0
; MSAA:  %5 = extractelement <4 x float> %1, i64 0
; MSAA:  %6 = insertelement <3 x float> %4, float %5, i64 1
; MSAA:  %7 = extractelement <4 x float> %2, i64 0
; MSAA:  %8 = insertelement <3 x float> %6, float %7, i64 2

; MSAA:  %9 = extractelement <4 x float> %0, i64 1
; MSAA:  %10 = insertelement <3 x float> undef, float  %9, i64 0
; MSAA:  %11 = extractelement <4 x float> %1, i64 1
; MSAA:  %12 = insertelement <3 x float> %10, float %11, i64 1
; MSAA:  %13 = extractelement <4 x float> %2, i64 1
; MSAA:  %14 = insertelement <3 x float> %12, float %13, i64 2

; MSAA:  %15 = extractelement <4 x float> %0, i64 2
; MSAA:  %16 = insertelement <3 x float> undef, float %15, i64 0
; MSAA:  %17 = extractelement <4 x float> %1, i64 2
; MSAA:  %18 = insertelement <3 x float> %16, float %17, i64 1
; MSAA:  %19 = extractelement <4 x float> %2, i64 2
; MSAA:  %20 = insertelement <3 x float> %18, float %19, i64 2

; MSAA:  %21 = extractelement <4 x float> %0, i64 3
; MSAA:  %22 = insertelement <3 x float> undef, float %21, i64 0
; MSAA:  %23 = extractelement <4 x float> %1, i64 3
; MSAA:  %24 = insertelement <3 x float> %22, float %23, i64 1
; MSAA:  %25 = extractelement <4 x float> %2, i64 3
; MSAA:  %26 = insertelement <3 x float> %24, float %25, i64 2
define amdgpu_ps [4 x <3 x float>] @load_2dmsaa_v4v3f32_dmask7(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <3 x float>] undef, <3 x float> %i, 0
  %i5 = insertvalue [4 x <3 x float>] %i4, <3 x float> %i1, 1
  %i6 = insertvalue [4 x <3 x float>] %i5, <3 x float> %i2, 2
  %i7 = insertvalue [4 x <3 x float>] %i6, <3 x float> %i3, 3
  ret [4 x <3 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4v3f32_dmask7_group1

; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 4, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 2, i32 %s, i32 %t, i32 4, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %2 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 4, <8 x i32> %rsrc, i32 0, i32 0)

; MSAA:  %3 = extractelement <4 x float> %0, i64 0
; MSAA:  %4 = insertelement <3 x float> undef, float %3, i64 0
; MSAA:  %5 = extractelement <4 x float> %1, i64 0
; MSAA:  %6 = insertelement <3 x float> %4, float %5, i64 1
; MSAA:  %7 = extractelement <4 x float> %2, i64 0
; MSAA:  %8 = insertelement <3 x float> %6, float %7, i64 2

; MSAA:  %9 = extractelement <4 x float> %0, i64 1
; MSAA:  %10 = insertelement <3 x float> undef, float  %9, i64 0
; MSAA:  %11 = extractelement <4 x float> %1, i64 1
; MSAA:  %12 = insertelement <3 x float> %10, float %11, i64 1
; MSAA:  %13 = extractelement <4 x float> %2, i64 1
; MSAA:  %14 = insertelement <3 x float> %12, float %13, i64 2

; MSAA:  %15 = extractelement <4 x float> %0, i64 2
; MSAA:  %16 = insertelement <3 x float> undef, float %15, i64 0
; MSAA:  %17 = extractelement <4 x float> %1, i64 2
; MSAA:  %18 = insertelement <3 x float> %16, float %17, i64 1
; MSAA:  %19 = extractelement <4 x float> %2, i64 2
; MSAA:  %20 = insertelement <3 x float> %18, float %19, i64 2

; MSAA:  %21 = extractelement <4 x float> %0, i64 3
; MSAA:  %22 = insertelement <3 x float> undef, float %21, i64 0
; MSAA:  %23 = extractelement <4 x float> %1, i64 3
; MSAA:  %24 = insertelement <3 x float> %22, float %23, i64 1
; MSAA:  %25 = extractelement <4 x float> %2, i64 3
; MSAA:  %26 = insertelement <3 x float> %24, float %25, i64 2
define amdgpu_ps [4 x <3 x float>] @load_2dmsaa_v4v3f32_dmask7_group1(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %i = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 4, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 5, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 6, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32 7, i32 %s, i32 %t, i32 7, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x <3 x float>] undef, <3 x float> %i, 0
  %i5 = insertvalue [4 x <3 x float>] %i4, <3 x float> %i1, 1
  %i6 = insertvalue [4 x <3 x float>] %i5, <3 x float> %i2, 2
  %i7 = insertvalue [4 x <3 x float>] %i6, <3 x float> %i3, 3
  ret [4 x <3 x float>] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4f32_sections
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  call void @llvm.amdgcn.image.store.2dmsaa.f32.i32(float %vdata, i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %3 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %4 = extractelement <4 x float> %3, i64 2
; MSAA:  %5 = extractelement <4 x float> %3, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_sections(<8 x i32> inreg %rsrc, float %vdata, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  call void @llvm.amdgcn.image.store.2dmsaa.f32.i32(float %vdata, i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  ret [4 x float] %i7
}

; MSAA-LABEL: @load_2dmsaa_v4f32_blocks
; MSAA:  %0 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %1 = extractelement <4 x float> %0, i64 0
; MSAA:  %2 = extractelement <4 x float> %0, i64 1
; MSAA:  %3 = extractelement <4 x float> %0, i64 2
; MSAA:  %4 = extractelement <4 x float> %0, i64 3
; MSAA-LABEL: if_equal:
; MSAA:  %5 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %6 = extractelement <4 x float> %5, i64 0
; MSAA:  %7 = extractelement <4 x float> %5, i64 1
; MSAA:  %8 = extractelement <4 x float> %5, i64 2
; MSAA:  %9 = extractelement <4 x float> %5, i64 3
; MSAA-LABEL: if_unequal:
; MSAA:  %10 = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
; MSAA:  %11 = extractelement <4 x float> %10, i64 0
; MSAA:  %12 = extractelement <4 x float> %10, i64 1
; MSAA:  %13 = extractelement <4 x float> %10, i64 2
; MSAA:  %14 = extractelement <4 x float> %10, i64 3
define amdgpu_ps [4 x float] @load_2dmsaa_v4f32_blocks(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %cond) {
main_body:
  %i = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i1 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i2 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i3 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i4 = insertvalue [4 x float] undef, float %i, 0
  %i5 = insertvalue [4 x float] %i4, float %i1, 1
  %i6 = insertvalue [4 x float] %i5, float %i2, 2
  %i7 = insertvalue [4 x float] %i6, float %i3, 3
  %i8 = trunc i32 %cond to i1
  br i1 %i8, label %if_equal, label %if_unequal
if_equal:
  %i9 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i10 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i11 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i12 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i13 = insertvalue [4 x float] undef, float %i9, 0
  %i14 = insertvalue [4 x float] %i13, float %i10, 1
  %i15 = insertvalue [4 x float] %i14, float %i11, 2
  %i16 = insertvalue [4 x float] %i15, float %i12, 3
  br label %merge
if_unequal:
  %i17 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 0, <8 x i32> %rsrc, i32 0, i32 0)
  %i18 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 1, <8 x i32> %rsrc, i32 0, i32 0)
  %i19 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 2, <8 x i32> %rsrc, i32 0, i32 0)
  %i20 = call float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32 1, i32 %s, i32 %t, i32 3, <8 x i32> %rsrc, i32 0, i32 0)
  %i21 = insertvalue [4 x float] undef, float %i17, 0
  %i22 = insertvalue [4 x float] %i21, float %i18, 1
  %i23 = insertvalue [4 x float] %i22, float %i19, 2
  %i24 = insertvalue [4 x float] %i23, float %i20, 3
  br label %merge
merge:
  %i25 = phi [4 x float] [%i16, %if_equal], [%i24, %if_unequal]
  ret [4 x float] %i25
}

declare float @llvm.amdgcn.image.load.2dmsaa.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare <2 x float> @llvm.amdgcn.image.load.2dmsaa.v2f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare <3 x float> @llvm.amdgcn.image.load.2dmsaa.v3f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0

declare float @llvm.amdgcn.image.load.2darraymsaa.f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare <2 x float> @llvm.amdgcn.image.load.2darraymsaa.v2f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0
declare <3 x float> @llvm.amdgcn.image.load.2darraymsaa.v3f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #0

declare half @llvm.amdgcn.image.load.2dmsaa.f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #0

declare void @llvm.amdgcn.image.store.2dmsaa.f32.i32(float, i32, i32, i32, i32, <8 x i32>, i32, i32)

attributes #0 = { nounwind readonly willreturn }
