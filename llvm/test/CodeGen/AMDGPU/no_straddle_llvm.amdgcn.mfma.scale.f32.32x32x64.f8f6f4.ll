;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -global-isel=0 -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx950 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis


define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp0(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_1_1__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 1, i32 %scale0, i32 1, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_2_2__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 2, i32 %scale0, i32 2, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_3_3__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 3, i32 %scale0, i32 3, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_3__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 3, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_3_0__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 3, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_2_3__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 2, i32 %scale0, i32 3, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_3_2__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 3, i32 %scale0, i32 2, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp0__constant_scale_0_0(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp1__constant_scale_0_0(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp2(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp2__constant_scale_0_0(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp3(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp3__constant_scale_0_0(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp4(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz0__blgp4__constant_scale_0_0(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp0(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp0__constant_scale_0_0(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}


define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp1__constant_scale_0_0(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp2(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp2__constant_scale_0(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp3(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp3__constant_scale_0_0(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp4(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz1__blgp4__constant_scale_0_0(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 1, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp0(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp0__constant_scale_0_0(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp1(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp1__constant_scale_0_0(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp2(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp2__constant_scale_0_0(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp3(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp3__constant_scale_0_0(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}


define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp0(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp0__constant_scale_0_0(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp1(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp1__constant_scale_0_0(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp2(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp2__constant_scale_0_0(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp4(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp4__constant_scale_0_0(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp3(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz3__blgp3__constant_scale_0_0(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 3, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp4(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz2__blgp4__constant_scale_0_0(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp0(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp0__constant_scale_0_0(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp1(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp1__constant_scale_0_0(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 1, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp2(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp2__constant_scale_0_0(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp3(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp3__constant_scale_0_0(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 3, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp4(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__cbsz4__blgp4__constant_scale_0_0(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}


define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__sgpr_scaleA__sgpr_scaleB(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 inreg %scale0, i32 inreg %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__sgpr_scaleA__vgpr_scaleB(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 inreg %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__vgpr_scaleA__sgpr_scaleB(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 inreg %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0_sgprs(<8 x i32> inreg %arg0, <8 x i32> inreg %arg1, <16 x float> inreg %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0_sgpr_vgpr_vgpr__sgpr_vgpr(<8 x i32> inreg %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 inreg %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0_sgpr_vgpr_vgpr__vgpr_sgpr(<8 x i32> inreg %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 inreg %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0_vgpr_sgpr_vgpr__vgpr_sgpr(<8 x i32> %arg0, <8 x i32> inreg %arg1, <16 x float> %arg2, i32 %scale0, i32 inreg %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0_vgpr_vgpr_sgpr__vgpr_sgpr(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> inreg %arg2, i32 %scale0, i32 inreg %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0_sgpr_vgpr_sgpr__vgpr_sgpr(<8 x i32> inreg %arg0, <8 x i32> %arg1, <16 x float> inreg %arg2, i32 %scale0, i32 inreg %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__scaleA_inlineimm__scaleB_inlineimm(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 2, i32 33, i32 2, i32 -2)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__scaleA_kimm__scaleB_inlineimm(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 2, i32 65, i32 2, i32 -2)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__scaleA_kimm__scaleB_FP_literal(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 2, i32 65, i32 2, i32 1065353216)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__scaleA_FP_literal__scaleB_inlineimm(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 2, i32 1065353216, i32 2, i32 -2)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__scaleA_FP_literal__scaleB_FP_literal(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 2, i32 1042479491, i32 2, i32 1065353216)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__scaleA_kimm__scaleB_kimm(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 2, i32 65, i32 2, i32 77)
  ret <16 x float> %result
}

define amdgpu_kernel void @test_mfma_scale_f32_32x32x64_f8f6f4__vgprcd(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1, ptr addrspace(1) %ptr) #0 {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 3, i32 %scale0, i32 1, i32 %scale1)
  store <16 x float> %result, ptr addrspace(1) %ptr, align 64
  ret void
}

define amdgpu_kernel void @test_mfma_scale_f32_32x32x64_f8f6f4__vgprcd___scaleA_kimm__scaleB__inlineimm(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, ptr addrspace(1) %ptr) #0 {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 3, i32 65, i32 1, i32 -2)
  store <16 x float> %result, ptr addrspace(1) %ptr, align 64
  ret void
}

define amdgpu_kernel void @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__nonmac(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) #1 {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
  store volatile <16 x float> %arg2, ptr addrspace(1) null, align 64
  store volatile <16 x float> %result, ptr addrspace(1) null, align 64
  ret void
}

define amdgpu_kernel void @test_mfma_scale_f32_32x32x64_f8f6f4_25_42__nonmac(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) #1 {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 0, i32 25, i32 0, i32 42)
  store volatile <16 x float> %arg2, ptr addrspace(1) null, align 64
  store volatile <16 x float> %result, ptr addrspace(1) null, align 64
  ret void
}

define amdgpu_kernel void @test_mfma_scale_f32_32x32x64_f8f6f4_0_0__vgprcd_nonmac(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) #0 {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0)
  store volatile <16 x float> %arg2, ptr addrspace(1) null, align 64
  store volatile <16 x float> %result, ptr addrspace(1) null, align 64
  ret void
}

define amdgpu_kernel void @test_mfma_scale_f32_32x32x64_f8f6f4_25_42__vgprcd_nonmac(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) #0 {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 0, i32 25, i32 0, i32 42)
  store volatile <16 x float> %arg2, ptr addrspace(1) null, align 64
  store volatile <16 x float> %result, ptr addrspace(1) null, align 64
  ret void
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___constant_scale_0_0_a(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___constant_scale_0_0_b(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 3, i32 0, i32 1, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___constant_scale_0_1(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___constant_scale_1_0_a(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0)
  ret <16 x float> %result
}


define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp8__v8i32_fp6(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp6__v8i32_fp8(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp6__v8i32_fp6(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp6__v8i32_fp6__0_scale(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 2, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp8__v8i32_fp4(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp4__v8i32_fp8(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp8__v6i32_fp4(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v6i32_fp4__v8i32_fp8(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp4__v8i32_fp4(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp4__v8i32_fp4__0_scale(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32>, <6 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32>, <6 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32>, <8 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32>, <4 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32>, <8 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32>, <4 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32>, <6 x i32>, <16 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2

attributes #0 = { "amdgpu-flat-work-group-size"="512,512" }
attributes #1 = { "amdgpu-flat-work-group-size"="128,128" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
