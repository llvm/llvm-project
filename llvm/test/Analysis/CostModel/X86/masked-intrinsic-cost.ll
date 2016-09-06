; RUN: opt -S -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -cost-model -analyze < %s | FileCheck %s --check-prefix=AVX2
; RUN: opt -S -mtriple=x86_64-apple-darwin -mcpu=knl -cost-model -analyze < %s | FileCheck %s --check-prefix=KNL
; RUN: opt -S -mtriple=x86_64-apple-darwin -mcpu=skx -cost-model -analyze < %s | FileCheck %s --check-prefix=SKX


; AVX2-LABEL: test1
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define <2 x double> @test1(<2 x i64> %trigger, <2 x double>* %addr, <2 x double> %dst) {
  %mask = icmp eq <2 x i64> %trigger, zeroinitializer
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %addr, i32 4, <2 x i1>%mask, <2 x double>%dst)
  ret <2 x double> %res
}

; AVX2-LABEL: test2
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define <4 x i32> @test2(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %dst) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %addr, i32 4, <4 x i1>%mask, <4 x i32>%dst)
  ret <4 x i32> %res
}

; AVX2-LABEL: test3
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define void @test3(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %val) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>%val, <4 x i32>* %addr, i32 4, <4 x i1>%mask)
  ret void
}

; AVX2-LABEL: test4
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define <8 x float> @test4(<8 x i32> %trigger, <8 x float>* %addr, <8 x float> %dst) {
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x float> @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %addr, i32 4, <8 x i1>%mask, <8 x float>%dst)
  ret <8 x float> %res
}

; AVX2-LABEL: test5
; AVX2: Found an estimated cost of 5 {{.*}}.masked
define void @test5(<2 x i32> %trigger, <2 x float>* %addr, <2 x float> %val) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v2f32.p0v2f32(<2 x float>%val, <2 x float>* %addr, i32 4, <2 x i1>%mask)
  ret void
}

; AVX2-LABEL: test6
; AVX2: Found an estimated cost of 6 {{.*}}.masked
define void @test6(<2 x i32> %trigger, <2 x i32>* %addr, <2 x i32> %val) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v2i32.p0v2i32(<2 x i32>%val, <2 x i32>* %addr, i32 4, <2 x i1>%mask)
  ret void
}

; AVX2-LABEL: test7
; AVX2: Found an estimated cost of 5 {{.*}}.masked
define <2 x float> @test7(<2 x i32> %trigger, <2 x float>* %addr, <2 x float> %dst) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x float> @llvm.masked.load.v2f32.p0v2f32(<2 x float>* %addr, i32 4, <2 x i1>%mask, <2 x float>%dst)
  ret <2 x float> %res
}

; AVX2-LABEL: test8
; AVX2: Found an estimated cost of 6 {{.*}}.masked
define <2 x i32> @test8(<2 x i32> %trigger, <2 x i32>* %addr, <2 x i32> %dst) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x i32> @llvm.masked.load.v2i32.p0v2i32(<2 x i32>* %addr, i32 4, <2 x i1>%mask, <2 x i32>%dst)
  ret <2 x i32> %res
}

define <2 x double> @test_gather_2f64(<2 x double*> %ptrs, <2 x i1> %mask, <2 x double> %src0)  {

; AVX2-LABEL: test_gather_2f64
; AVX2: Found an estimated cost of 7 {{.*}}.gather

; KNL-LABEL: test_gather_2f64
; KNL: Found an estimated cost of 7 {{.*}}.gather

; SKX-LABEL: test_gather_2f64
; SKX: Found an estimated cost of 7 {{.*}}.gather

%res = call <2 x double> @llvm.masked.gather.v2f64(<2 x double*> %ptrs, i32 4, <2 x i1> %mask, <2 x double> %src0)
  ret <2 x double> %res
}
declare <2 x double> @llvm.masked.gather.v2f64(<2 x double*> %ptrs, i32, <2 x i1> %mask, <2 x double> %src0)

define <4 x i32> @test_gather_4i32(<4 x i32*> %ptrs, <4 x i1> %mask, <4 x i32> %src0)  {

; AVX2-LABEL: test_gather_4i32
; AVX2: Found an estimated cost of 16 {{.*}}.gather

; KNL-LABEL: test_gather_4i32
; KNL: Found an estimated cost of 16 {{.*}}.gather

; SKX-LABEL: test_gather_4i32
; SKX: Found an estimated cost of 6 {{.*}}.gather

%res = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ptrs, i32 4, <4 x i1> %mask, <4 x i32> %src0)
  ret <4 x i32> %res
}

define <4 x i32> @test_gather_4i32_const_mask(<4 x i32*> %ptrs, <4 x i32> %src0)  {

; AVX2-LABEL: test_gather_4i32_const_mask
; AVX2: Found an estimated cost of 8 {{.*}}.gather

; KNL-LABEL: test_gather_4i32_const_mask
; KNL: Found an estimated cost of 8 {{.*}}.gather

; SKX-LABEL: test_gather_4i32_const_mask
; SKX: Found an estimated cost of 6 {{.*}}.gather

%res = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ptrs, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> %src0)
  ret <4 x i32> %res
}
declare <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ptrs, i32, <4 x i1> %mask, <4 x i32> %src0)

define <16 x float> @test_gather_16f32_const_mask(float* %base, <16 x i32> %ind) {

; AVX2-LABEL: test_gather_16f32_const_mask
; AVX2: Found an estimated cost of 30 {{.*}}.gather

; KNL-LABEL: test_gather_16f32_const_mask
; KNL: Found an estimated cost of 18 {{.*}}.gather

; SKX-LABEL: test_gather_16f32_const_mask
; SKX: Found an estimated cost of 18 {{.*}}.gather

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.v = getelementptr float, float* %base, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.v, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

define <16 x float> @test_gather_16f32_var_mask(float* %base, <16 x i32> %ind, <16 x i1>%mask) {

; AVX2-LABEL: test_gather_16f32_var_mask
; AVX2: Found an estimated cost of 62 {{.*}}.gather

; KNL-LABEL: test_gather_16f32_var_mask
; KNL: Found an estimated cost of 18 {{.*}}.gather

; SKX-LABEL: test_gather_16f32_var_mask
; SKX: Found an estimated cost of 18 {{.*}}.gather

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.v = getelementptr float, float* %base, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.v, i32 4, <16 x i1> %mask, <16 x float> undef)
  ret <16 x float>%res
}

define <16 x float> @test_gather_16f32_ra_var_mask(<16 x float*> %ptrs, <16 x i32> %ind, <16 x i1>%mask) {

; AVX2-LABEL: test_gather_16f32_ra_var_mask
; AVX2: Found an estimated cost of 62 {{.*}}.gather

; KNL-LABEL: test_gather_16f32_ra_var_mask
; KNL: Found an estimated cost of 20 {{.*}}.gather

; SKX-LABEL: test_gather_16f32_ra_var_mask
; SKX: Found an estimated cost of 20 {{.*}}.gather

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.v = getelementptr float, <16 x float*> %ptrs, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.v, i32 4, <16 x i1> %mask, <16 x float> undef)
  ret <16 x float>%res
}

define <16 x float> @test_gather_16f32_const_mask2(float* %base, <16 x i32> %ind) {

; AVX2-LABEL: test_gather_16f32_const_mask2
; AVX2: Found an estimated cost of 30 {{.*}}.gather

; KNL-LABEL: test_gather_16f32_const_mask2
; KNL: Found an estimated cost of 18 {{.*}}.gather

; SKX-LABEL: test_gather_16f32_const_mask2
; SKX: Found an estimated cost of 18 {{.*}}.gather

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

define void @test_scatter_16i32(i32* %base, <16 x i32> %ind, i16 %mask, <16 x i32>%val) {
; AVX2-LABEL: test_scatter_16i32
; AVX2: Found an estimated cost of 64 {{.*}}.scatter

; KNL-LABEL: test_scatter_16i32
; KNL: Found an estimated cost of 18 {{.*}}.scatter

; SKX-LABEL: test_scatter_16i32
; SKX: Found an estimated cost of 18 {{.*}}.scatter

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  call void @llvm.masked.scatter.v16i32(<16 x i32>%val, <16 x i32*> %gep.random, i32 4, <16 x i1> %imask)
  ret void
}

define void @test_scatter_8i32(<8 x i32>%a1, <8 x i32*> %ptr, <8 x i1>%mask) {
; AVX2-LABEL: test_scatter_8i32
; AVX2: Found an estimated cost of 32 {{.*}}.scatter

; KNL-LABEL: test_scatter_8i32
; KNL: Found an estimated cost of 10 {{.*}}.scatter

; SKX-LABEL: test_scatter_8i32
; SKX: Found an estimated cost of 10 {{.*}}.scatter

  call void @llvm.masked.scatter.v8i32(<8 x i32> %a1, <8 x i32*> %ptr, i32 4, <8 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.v8i32(<8 x i32> %a1, <8 x i32*> %ptr, i32, <8 x i1> %mask)

define void @test_scatter_4i32(<4 x i32>%a1, <4 x i32*> %ptr, <4 x i1>%mask) {
; AVX2-LABEL: test_scatter_4i32
; AVX2: Found an estimated cost of 16 {{.*}}.scatter

; KNL-LABEL: test_scatter_4i32
; KNL: Found an estimated cost of 16 {{.*}}.scatter

; SKX-LABEL: test_scatter_4i32
; SKX: Found an estimated cost of 6 {{.*}}.scatter

  call void @llvm.masked.scatter.v4i32(<4 x i32> %a1, <4 x i32*> %ptr, i32 4, <4 x i1> %mask)
  ret void
}

define <4 x float> @test_gather_4f32(float* %ptr, <4 x i32> %ind, <4 x i1>%mask) {

; AVX2-LABEL: test_gather_4f32
; AVX2: Found an estimated cost of 15 {{.*}}.gather

; KNL-LABEL: test_gather_4f32
; KNL: Found an estimated cost of 15 {{.*}}.gather

; SKX-LABEL: test_gather_4f32
; SKX: Found an estimated cost of 6 {{.*}}.gather

  %sext_ind = sext <4 x i32> %ind to <4 x i64>
  %gep.v = getelementptr float, float* %ptr, <4 x i64> %sext_ind

  %res = call <4 x float> @llvm.masked.gather.v4f32(<4 x float*> %gep.v, i32 4, <4 x i1> %mask, <4 x float> undef)
  ret <4 x float>%res
}

define <4 x float> @test_gather_4f32_const_mask(float* %ptr, <4 x i32> %ind) {

; AVX2-LABEL: test_gather_4f32_const_mask
; AVX2: Found an estimated cost of 7 {{.*}}.gather

; KNL-LABEL: test_gather_4f32_const_mask
; KNL: Found an estimated cost of 7 {{.*}}.gather

; SKX-LABEL: test_gather_4f32_const_mask
; SKX: Found an estimated cost of 6 {{.*}}.gather

  %sext_ind = sext <4 x i32> %ind to <4 x i64>
  %gep.v = getelementptr float, float* %ptr, <4 x i64> %sext_ind

  %res = call <4 x float> @llvm.masked.gather.v4f32(<4 x float*> %gep.v, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x float> undef)
  ret <4 x float>%res
}

declare <4 x float> @llvm.masked.gather.v4f32(<4 x float*> %gep.v, i32, <4 x i1> %mask, <4 x float> )
declare void @llvm.masked.scatter.v4i32(<4 x i32> %a1, <4 x i32*> %ptr, i32, <4 x i1> %mask)
declare void @llvm.masked.scatter.v16i32(<16 x i32>%val, <16 x i32*> %gep.random, i32, <16 x i1> %imask)
declare <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.v, i32, <16 x i1> %mask, <16 x float>)

declare <16 x i32> @llvm.masked.load.v16i32.p0v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)
declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)
declare <2 x i32> @llvm.masked.load.v2i32.p0v2i32(<2 x i32>*, i32, <2 x i1>, <2 x i32>)
declare void @llvm.masked.store.v16i32.p0v16i32(<16 x i32>, <16 x i32>*, i32, <16 x i1>)
declare void @llvm.masked.store.v8i32.p0v8i32(<8 x i32>, <8 x i32>*, i32, <8 x i1>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)
declare void @llvm.masked.store.v2f32.p0v2f32(<2 x float>, <2 x float>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2i32.p0v2i32(<2 x i32>, <2 x i32>*, i32, <2 x i1>)
declare void @llvm.masked.store.v16f32.p0v16f32(<16 x float>, <16 x float>*, i32, <16 x i1>)
declare <16 x float> @llvm.masked.load.v16f32.p0v16f32(<16 x float>*, i32, <16 x i1>, <16 x float>)
declare <8 x float> @llvm.masked.load.v8f32.p0v8f32(<8 x float>*, i32, <8 x i1>, <8 x float>)
declare <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>)
declare <2 x float> @llvm.masked.load.v2f32.p0v2f32(<2 x float>*, i32, <2 x i1>, <2 x float>)
declare <8 x double> @llvm.masked.load.v8f64.p0v8f64(<8 x double>*, i32, <8 x i1>, <8 x double>)
declare <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>*, i32, <4 x i1>, <4 x double>)
declare <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>*, i32, <2 x i1>, <2 x double>)
declare void @llvm.masked.store.v8f64.p0v8f64(<8 x double>, <8 x double>*, i32, <8 x i1>)
declare void @llvm.masked.store.v2f64.p0v2f64(<2 x double>, <2 x double>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2i64.p0v2i64(<2 x i64>, <2 x i64>*, i32, <2 x i1>)
