; RUN: llc -mtriple=x86_64-apple-darwin  -mcpu=knl < %s | FileCheck %s -check-prefix=KNL
; RUN: opt -mtriple=x86_64-apple-darwin -codegenprepare -mcpu=corei7-avx -S < %s | FileCheck %s -check-prefix=SCALAR


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; KNL-LABEL: test1
; KNL: kxnorw  %k1, %k1, %k1
; KNL: vgatherdps      (%rdi,%zmm0,4), %zmm1 {%k1}

; SCALAR-LABEL: test1
; SCALAR:      extractelement <16 x float*> 
; SCALAR-NEXT: load float
; SCALAR-NEXT: insertelement <16 x float>
; SCALAR-NEXT: extractelement <16 x float*>
; SCALAR-NEXT: load float

define <16 x float> @test1(float* %base, <16 x i32> %ind) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, <16 x i64> %sext_ind
  
  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

declare <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*>, i32, <16 x i1>, <16 x i32>)
declare <16 x float> @llvm.masked.gather.v16f32(<16 x float*>, i32, <16 x i1>, <16 x float>)
declare <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> , i32, <8 x i1> , <8 x i32> )
  
; KNL-LABEL: test2
; KNL: kmovw %esi, %k1
; KNL: vgatherdps      (%rdi,%zmm0,4), %zmm1 {%k1}

; SCALAR-LABEL: test2
; SCALAR:      extractelement <16 x float*> 
; SCALAR-NEXT: load float
; SCALAR-NEXT: insertelement <16 x float>
; SCALAR-NEXT: br label %else
; SCALAR: else:
; SCALAR-NEXT:  %res.phi.else = phi 
; SCALAR-NEXT:  %Mask1 = extractelement <16 x i1> %imask, i32 1
; SCALAR-NEXT:  %ToLoad1 = icmp eq i1 %Mask1, true
; SCALAR-NEXT:  br i1 %ToLoad1, label %cond.load1, label %else2

define <16 x float> @test2(float* %base, <16 x i32> %ind, i16 %mask) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, <16 x i64> %sext_ind
  %imask = bitcast i16 %mask to <16 x i1>
  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> %imask, <16 x float>undef)
  ret <16 x float> %res
}

; KNL-LABEL: test3
; KNL: kmovw %esi, %k1
; KNL: vpgatherdd      (%rdi,%zmm0,4), %zmm1 {%k1}
define <16 x i32> @test3(i32* %base, <16 x i32> %ind, i16 %mask) {

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i64> %sext_ind
  %imask = bitcast i16 %mask to <16 x i1>
  %res = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  ret <16 x i32> %res
}

; KNL-LABEL: test4
; KNL: kmovw %esi, %k1
; KNL: kmovw
; KNL: vpgatherdd
; KNL: vpgatherdd

define <16 x i32> @test4(i32* %base, <16 x i32> %ind, i16 %mask) {

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  %gt1 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  %gt2 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>%gt1)
  %res = add <16 x i32> %gt1, %gt2
  ret <16 x i32> %res
}

; KNL-LABEL: test5
; KNL: kmovw %k1, %k2
; KNL: vpscatterdd {{.*}}%k2
; KNL: vpscatterdd {{.*}}%k1

; SCALAR-LABEL: test5
; SCALAR:        %Mask0 = extractelement <16 x i1> %imask, i32 0
; SCALAR-NEXT:   %ToStore0 = icmp eq i1 %Mask0, true
; SCALAR-NEXT:   br i1 %ToStore0, label %cond.store, label %else
; SCALAR: cond.store:
; SCALAR-NEXT:  %Elt0 = extractelement <16 x i32> %val, i32 0
; SCALAR-NEXT:  %Ptr0 = extractelement <16 x i32*> %gep.random, i32 0
; SCALAR-NEXT:  store i32 %Elt0, i32* %Ptr0, align 4
; SCALAR-NEXT:  br label %else
; SCALAR: else:
; SCALAR-NEXT: %Mask1 = extractelement <16 x i1> %imask, i32 1
; SCALAR-NEXT:  %ToStore1 = icmp eq i1 %Mask1, true
; SCALAR-NEXT:  br i1 %ToStore1, label %cond.store1, label %else2

define void @test5(i32* %base, <16 x i32> %ind, i16 %mask, <16 x i32>%val) {

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  call void @llvm.masked.scatter.v16i32(<16 x i32>%val, <16 x i32*> %gep.random, i32 4, <16 x i1> %imask)
  call void @llvm.masked.scatter.v16i32(<16 x i32>%val, <16 x i32*> %gep.random, i32 4, <16 x i1> %imask)
  ret void
}

declare void @llvm.masked.scatter.v8i32(<8 x i32> , <8 x i32*> , i32 , <8 x i1> )
declare void @llvm.masked.scatter.v16i32(<16 x i32> , <16 x i32*> , i32 , <16 x i1> )

; KNL-LABEL: test6
; KNL: kxnorw  %k1, %k1, %k1
; KNL: kxnorw  %k2, %k2, %k2
; KNL: vpgatherqd      (,%zmm{{.*}}), %ymm{{.*}} {%k2}
; KNL: vpscatterqd     %ymm{{.*}}, (,%zmm{{.*}}) {%k1}

; SCALAR-LABEL: test6
; SCALAR:        store i32 %Elt0, i32* %Ptr01, align 4
; SCALAR-NEXT:   %Elt1 = extractelement <8 x i32> %a1, i32 1
; SCALAR-NEXT:   %Ptr12 = extractelement <8 x i32*> %ptr, i32 1
; SCALAR-NEXT:   store i32 %Elt1, i32* %Ptr12, align 4
; SCALAR-NEXT:   %Elt2 = extractelement <8 x i32> %a1, i32 2
; SCALAR-NEXT:   %Ptr23 = extractelement <8 x i32*> %ptr, i32 2
; SCALAR-NEXT:   store i32 %Elt2, i32* %Ptr23, align 4

define <8 x i32> @test6(<8 x i32>%a1, <8 x i32*> %ptr) {

  %a = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %ptr, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)

  call void @llvm.masked.scatter.v8i32(<8 x i32> %a1, <8 x i32*> %ptr, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret <8 x i32>%a
}

; In this case the index should be promoted to <8 x i64> for KNL
; KNL-LABEL: test7
; KNL: vpmovsxdq %ymm0, %zmm0
; KNL: kmovw   %k1, %k2
; KNL: vpgatherqd {{.*}} {%k2}
; KNL: vpgatherqd {{.*}} {%k1}
define <8 x i32> @test7(i32* %base, <8 x i32> %ind, i8 %mask) {

  %broadcast.splatinsert = insertelement <8 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <8 x i32*> %broadcast.splatinsert, <8 x i32*> undef, <8 x i32> zeroinitializer

  %gep.random = getelementptr i32, <8 x i32*> %broadcast.splat, <8 x i32> %ind
  %imask = bitcast i8 %mask to <8 x i1>
  %gt1 = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %gep.random, i32 4, <8 x i1> %imask, <8 x i32>undef)
  %gt2 = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %gep.random, i32 4, <8 x i1> %imask, <8 x i32>%gt1)
  %res = add <8 x i32> %gt1, %gt2
  ret <8 x i32> %res
}

; No uniform base in this case, index <8 x i64> contains addresses,
; each gather call will be split into two
; KNL-LABEL: test8
; KNL: kshiftrw        $8, %k1, %k2
; KNL: vpgatherqd
; KNL: vpgatherqd
; KNL: vinserti64x4
; KNL: vpgatherqd
; KNL: vpgatherqd
; KNL: vinserti64x4
define <16 x i32> @test8(<16 x i32*> %ptr.random, <16 x i32> %ind, i16 %mask) {
  %imask = bitcast i16 %mask to <16 x i1>
  %gt1 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %ptr.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  %gt2 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %ptr.random, i32 4, <16 x i1> %imask, <16 x i32>%gt1)
  %res = add <16 x i32> %gt1, %gt2
  ret <16 x i32> %res
}

%struct.RT = type { i8, [10 x [20 x i32]], i8 }
%struct.ST = type { i32, double, %struct.RT }

; Masked gather for agregate types
; Test9 and Test10 should give the same result (scalar and vector indices in GEP)

; KNL-LABEL: test9
; KNL: vpbroadcastq    %rdi, %zmm
; KNL: vpmovsxdq
; KNL: vpbroadcastq
; KNL: vpmuludq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpgatherqd      (,%zmm

define <8 x i32> @test9(%struct.ST* %base, <8 x i64> %ind1, <8 x i32>%ind5) {
entry:
  %broadcast.splatinsert = insertelement <8 x %struct.ST*> undef, %struct.ST* %base, i32 0
  %broadcast.splat = shufflevector <8 x %struct.ST*> %broadcast.splatinsert, <8 x %struct.ST*> undef, <8 x i32> zeroinitializer

  %arrayidx = getelementptr  %struct.ST, <8 x %struct.ST*> %broadcast.splat, <8 x i64> %ind1, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>, <8 x i32><i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32> %ind5, <8 x i64> <i64 13, i64 13, i64 13, i64 13, i64 13, i64 13, i64 13, i64 13>
  %res = call <8 x i32 >  @llvm.masked.gather.v8i32(<8 x i32*>%arrayidx, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  ret <8 x i32> %res
}

; KNL-LABEL: test10
; KNL: vpbroadcastq    %rdi, %zmm
; KNL: vpmovsxdq
; KNL: vpbroadcastq
; KNL: vpmuludq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpgatherqd      (,%zmm
define <8 x i32> @test10(%struct.ST* %base, <8 x i64> %i1, <8 x i32>%ind5) {
entry:
  %broadcast.splatinsert = insertelement <8 x %struct.ST*> undef, %struct.ST* %base, i32 0
  %broadcast.splat = shufflevector <8 x %struct.ST*> %broadcast.splatinsert, <8 x %struct.ST*> undef, <8 x i32> zeroinitializer

  %arrayidx = getelementptr  %struct.ST, <8 x %struct.ST*> %broadcast.splat, <8 x i64> %i1, i32 2, i32 1, <8 x i32> %ind5, i64 13
  %res = call <8 x i32 >  @llvm.masked.gather.v8i32(<8 x i32*>%arrayidx, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  ret <8 x i32> %res
}

; Splat index in GEP, requires broadcast
; KNL-LABEL: test11
; KNL: vpbroadcastd    %esi, %zmm
; KNL: vgatherdps      (%rdi,%zmm
define <16 x float> @test11(float* %base, i32 %ind) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, i32 %ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

; We are checking the uniform base here. It is taken directly from input to vgatherdps
; KNL-LABEL: test12
; KNL: kxnorw  %k1, %k1, %k1
; KNL: vgatherdps      (%rdi,%zmm
define <16 x float> @test12(float* %base, <16 x i32> %ind) {

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, float *%base, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

; The same as the previous, but the mask is undefined
; KNL-LABEL: test13
; KNL-NOT: kxnorw
; KNL: vgatherdps      (%rdi,%zmm
define <16 x float> @test13(float* %base, <16 x i32> %ind) {

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, float *%base, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> undef, <16 x float> undef)
  ret <16 x float>%res
}

; The base pointer is not splat, can't find unform base
; KNL-LABEL: test14
; KNL: vgatherqps      (,%zmm0)
; KNL: vgatherqps      (,%zmm0)
define <16 x float> @test14(float* %base, i32 %ind, <16 x float*> %vec) {

  %broadcast.splatinsert = insertelement <16 x float*> %vec, float* %base, i32 1
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, i32 %ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> undef, <16 x float> undef)
  ret <16 x float>%res
}


; KNL-LABEL: test15
; KNL: kmovw %eax, %k1
; KNL: vgatherdps      (%rdi,%zmm0,4), %zmm1 {%k1}

; SCALAR-LABEL: test15
; SCALAR:      extractelement <16 x float*> 
; SCALAR-NEXT: load float
; SCALAR-NEXT: insertelement <16 x float>
; SCALAR-NEXT: extractelement <16 x float*>
; SCALAR-NEXT: load float

define <16 x float> @test15(float* %base, <16 x i32> %ind) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 false, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <16 x float> undef)
  ret <16 x float>%res
}

; Check non-power-of-2 case. It should be scalarized.
declare <3 x i32> @llvm.masked.gather.v3i32(<3 x i32*>, i32, <3 x i1>, <3 x i32>)
; KNL-LABEL: test16
; KNL: testb
; KNL: je
; KNL: testb
; KNL: je
; KNL: testb
; KNL: je
define <3 x i32> @test16(<3 x i32*> %base, <3 x i32> %ind, <3 x i1> %mask, <3 x i32> %src0) {
  %sext_ind = sext <3 x i32> %ind to <3 x i64>
  %gep.random = getelementptr i32, <3 x i32*> %base, <3 x i64> %sext_ind
  %res = call <3 x i32> @llvm.masked.gather.v3i32(<3 x i32*> %gep.random, i32 4, <3 x i1> %mask, <3 x i32> %src0)
  ret <3 x i32>%res
}

declare <16 x float*> @llvm.masked.gather.v16p0f32(<16 x float**>, i32, <16 x i1>, <16 x float*>)

; KNL-LABEL: test17
; KNL: vpgatherqq
; KNL: vpgatherqq
define <16 x float*> @test17(<16 x float**> %ptrs) {

  %res = call <16 x float*> @llvm.masked.gather.v16p0f32(<16 x float**> %ptrs, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float*> undef)
  ret <16 x float*>%res
}
