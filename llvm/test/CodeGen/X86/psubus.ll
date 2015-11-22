; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse2 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

define void @test1(i16* nocapture %head) nounwind {
; SSE-LABEL: test1:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    psubusw {{.*}}(%rip), %xmm0
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: test1:
; AVX:       ## BB#0: ## %vector.ph
; AVX-NEXT:    vmovdqu (%rdi), %xmm0
; AVX-NEXT:    vpsubusw {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <8 x i16>*
  %2 = load <8 x i16>, <8 x i16>* %1, align 2
  %3 = icmp slt <8 x i16> %2, zeroinitializer
  %4 = xor <8 x i16> %2, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %5 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> zeroinitializer
  store <8 x i16> %5, <8 x i16>* %1, align 2
  ret void
}

define void @test2(i16* nocapture %head) nounwind {
; SSE-LABEL: test2:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    psubusw {{.*}}(%rip), %xmm0
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: test2:
; AVX:       ## BB#0: ## %vector.ph
; AVX-NEXT:    vmovdqu (%rdi), %xmm0
; AVX-NEXT:    vpsubusw {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <8 x i16>*
  %2 = load <8 x i16>, <8 x i16>* %1, align 2
  %3 = icmp ugt <8 x i16> %2, <i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766>
  %4 = add <8 x i16> %2, <i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767>
  %5 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> zeroinitializer
  store <8 x i16> %5, <8 x i16>* %1, align 2
  ret void
}

define void @test3(i16* nocapture %head, i16 zeroext %w) nounwind {
; SSE2-LABEL: test3:
; SSE2:       ## BB#0: ## %vector.ph
; SSE2-NEXT:    movd %esi, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,1,0,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    movdqu (%rdi), %xmm1
; SSE2-NEXT:    psubusw %xmm0, %xmm1
; SSE2-NEXT:    movdqu %xmm1, (%rdi)
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: test3:
; SSSE3:       ## BB#0: ## %vector.ph
; SSSE3-NEXT:    movd %esi, %xmm0
; SSSE3-NEXT:    pshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; SSSE3-NEXT:    movdqu (%rdi), %xmm1
; SSSE3-NEXT:    psubusw %xmm0, %xmm1
; SSSE3-NEXT:    movdqu %xmm1, (%rdi)
; SSSE3-NEXT:    retq
;
; AVX1-LABEL: test3:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovd %esi, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vmovdqu (%rdi), %xmm1
; AVX1-NEXT:    vpsubusw %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test3:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovd %esi, %xmm0
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm0
; AVX2-NEXT:    vmovdqu (%rdi), %xmm1
; AVX2-NEXT:    vpsubusw %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX2-NEXT:    retq
vector.ph:
  %0 = insertelement <8 x i16> undef, i16 %w, i32 0
  %broadcast15 = shufflevector <8 x i16> %0, <8 x i16> undef, <8 x i32> zeroinitializer
  %1 = getelementptr inbounds i16, i16* %head, i64 0
  %2 = bitcast i16* %1 to <8 x i16>*
  %3 = load <8 x i16>, <8 x i16>* %2, align 2
  %4 = icmp ult <8 x i16> %3, %broadcast15
  %5 = sub <8 x i16> %3, %broadcast15
  %6 = select <8 x i1> %4, <8 x i16> zeroinitializer, <8 x i16> %5
  store <8 x i16> %6, <8 x i16>* %2, align 2
  ret void
}

define void @test4(i8* nocapture %head) nounwind {
; SSE-LABEL: test4:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    psubusb {{.*}}(%rip), %xmm0
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: test4:
; AVX:       ## BB#0: ## %vector.ph
; AVX-NEXT:    vmovdqu (%rdi), %xmm0
; AVX-NEXT:    vpsubusb {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <16 x i8>*
  %2 = load <16 x i8>, <16 x i8>* %1, align 1
  %3 = icmp slt <16 x i8> %2, zeroinitializer
  %4 = xor <16 x i8> %2, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %5 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> zeroinitializer
  store <16 x i8> %5, <16 x i8>* %1, align 1
  ret void
}

define void @test5(i8* nocapture %head) nounwind {
; SSE-LABEL: test5:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    psubusb {{.*}}(%rip), %xmm0
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: test5:
; AVX:       ## BB#0: ## %vector.ph
; AVX-NEXT:    vmovdqu (%rdi), %xmm0
; AVX-NEXT:    vpsubusb {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <16 x i8>*
  %2 = load <16 x i8>, <16 x i8>* %1, align 1
  %3 = icmp ugt <16 x i8> %2, <i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126>
  %4 = add <16 x i8> %2, <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
  %5 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> zeroinitializer
  store <16 x i8> %5, <16 x i8>* %1, align 1
  ret void
}

define void @test6(i8* nocapture %head, i8 zeroext %w) nounwind {
; SSE2-LABEL: test6:
; SSE2:       ## BB#0: ## %vector.ph
; SSE2-NEXT:    movd %esi, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,1,0,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    movdqu (%rdi), %xmm1
; SSE2-NEXT:    psubusb %xmm0, %xmm1
; SSE2-NEXT:    movdqu %xmm1, (%rdi)
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: test6:
; SSSE3:       ## BB#0: ## %vector.ph
; SSSE3-NEXT:    movd %esi, %xmm0
; SSSE3-NEXT:    pxor %xmm1, %xmm1
; SSSE3-NEXT:    pshufb %xmm1, %xmm0
; SSSE3-NEXT:    movdqu (%rdi), %xmm1
; SSSE3-NEXT:    psubusb %xmm0, %xmm1
; SSSE3-NEXT:    movdqu %xmm1, (%rdi)
; SSSE3-NEXT:    retq
;
; AVX1-LABEL: test6:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovd %esi, %xmm0
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vmovdqu (%rdi), %xmm1
; AVX1-NEXT:    vpsubusb %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test6:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovd %esi, %xmm0
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm0
; AVX2-NEXT:    vmovdqu (%rdi), %xmm1
; AVX2-NEXT:    vpsubusb %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vmovdqu %xmm0, (%rdi)
; AVX2-NEXT:    retq
vector.ph:
  %0 = insertelement <16 x i8> undef, i8 %w, i32 0
  %broadcast15 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  %1 = getelementptr inbounds i8, i8* %head, i64 0
  %2 = bitcast i8* %1 to <16 x i8>*
  %3 = load <16 x i8>, <16 x i8>* %2, align 1
  %4 = icmp ult <16 x i8> %3, %broadcast15
  %5 = sub <16 x i8> %3, %broadcast15
  %6 = select <16 x i1> %4, <16 x i8> zeroinitializer, <16 x i8> %5
  store <16 x i8> %6, <16 x i8>* %2, align 1
  ret void
}

define void @test7(i16* nocapture %head) nounwind {
; SSE-LABEL: test7:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [32768,32768,32768,32768,32768,32768,32768,32768]
; SSE-NEXT:    psubusw %xmm2, %xmm0
; SSE-NEXT:    psubusw %xmm2, %xmm1
; SSE-NEXT:    movdqu %xmm1, 16(%rdi)
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: test7:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovups (%rdi), %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpgtw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpcmpgtw %xmm0, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; AVX1-NEXT:    vxorps {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm0, %ymm1, %ymm0
; AVX1-NEXT:    vmovups %ymm0, (%rdi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test7:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovdqu (%rdi), %ymm0
; AVX2-NEXT:    vpsubusw {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <16 x i16>*
  %2 = load <16 x i16>, <16 x i16>* %1, align 2
  %3 = icmp slt <16 x i16> %2, zeroinitializer
  %4 = xor <16 x i16> %2, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %5 = select <16 x i1> %3, <16 x i16> %4, <16 x i16> zeroinitializer
  store <16 x i16> %5, <16 x i16>* %1, align 2
  ret void
}

define void @test8(i16* nocapture %head) nounwind {
; SSE-LABEL: test8:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [32767,32767,32767,32767,32767,32767,32767,32767]
; SSE-NEXT:    psubusw %xmm2, %xmm0
; SSE-NEXT:    psubusw %xmm2, %xmm1
; SSE-NEXT:    movdqu %xmm1, 16(%rdi)
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: test8:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovups (%rdi), %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovaps {{.*#+}} xmm2 = [32768,32768,32768,32768,32768,32768,32768,32768]
; AVX1-NEXT:    vxorps %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [65534,65534,65534,65534,65534,65534,65534,65534]
; AVX1-NEXT:    vpcmpgtw %xmm4, %xmm3, %xmm3
; AVX1-NEXT:    vxorps %xmm2, %xmm0, %xmm2
; AVX1-NEXT:    vpcmpgtw %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [32769,32769,32769,32769,32769,32769,32769,32769]
; AVX1-NEXT:    vpaddw %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpaddw %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    vmovups %ymm0, (%rdi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test8:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovdqu (%rdi), %ymm0
; AVX2-NEXT:    vpsubusw {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <16 x i16>*
  %2 = load <16 x i16>, <16 x i16>* %1, align 2
  %3 = icmp ugt <16 x i16> %2, <i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766>
  %4 = add <16 x i16> %2, <i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767>
  %5 = select <16 x i1> %3, <16 x i16> %4, <16 x i16> zeroinitializer
  store <16 x i16> %5, <16 x i16>* %1, align 2
  ret void

}

define void @test9(i16* nocapture %head, i16 zeroext %w) nounwind {
; SSE2-LABEL: test9:
; SSE2:       ## BB#0: ## %vector.ph
; SSE2-NEXT:    movd %esi, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,1,0,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    movdqu (%rdi), %xmm1
; SSE2-NEXT:    movdqu 16(%rdi), %xmm2
; SSE2-NEXT:    psubusw %xmm0, %xmm1
; SSE2-NEXT:    psubusw %xmm0, %xmm2
; SSE2-NEXT:    movdqu %xmm2, 16(%rdi)
; SSE2-NEXT:    movdqu %xmm1, (%rdi)
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: test9:
; SSSE3:       ## BB#0: ## %vector.ph
; SSSE3-NEXT:    movd %esi, %xmm0
; SSSE3-NEXT:    pshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; SSSE3-NEXT:    movdqu (%rdi), %xmm1
; SSSE3-NEXT:    movdqu 16(%rdi), %xmm2
; SSSE3-NEXT:    psubusw %xmm0, %xmm1
; SSSE3-NEXT:    psubusw %xmm0, %xmm2
; SSSE3-NEXT:    movdqu %xmm2, 16(%rdi)
; SSSE3-NEXT:    movdqu %xmm1, (%rdi)
; SSSE3-NEXT:    retq
;
; AVX1-LABEL: test9:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovups (%rdi), %ymm0
; AVX1-NEXT:    vmovd %esi, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpsubw %xmm1, %xmm2, %xmm3
; AVX1-NEXT:    vpsubw %xmm1, %xmm0, %xmm4
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm4, %ymm3
; AVX1-NEXT:    vpmaxuw %xmm1, %xmm2, %xmm4
; AVX1-NEXT:    vpcmpeqw %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vpmaxuw %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpcmpeqw %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm3, %ymm0, %ymm0
; AVX1-NEXT:    vmovups %ymm0, (%rdi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test9:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovd %esi, %xmm0
; AVX2-NEXT:    vpbroadcastw %xmm0, %ymm0
; AVX2-NEXT:    vmovdqu (%rdi), %ymm1
; AVX2-NEXT:    vpsubusw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
vector.ph:
  %0 = insertelement <16 x i16> undef, i16 %w, i32 0
  %broadcast15 = shufflevector <16 x i16> %0, <16 x i16> undef, <16 x i32> zeroinitializer
  %1 = getelementptr inbounds i16, i16* %head, i64 0
  %2 = bitcast i16* %1 to <16 x i16>*
  %3 = load <16 x i16>, <16 x i16>* %2, align 2
  %4 = icmp ult <16 x i16> %3, %broadcast15
  %5 = sub <16 x i16> %3, %broadcast15
  %6 = select <16 x i1> %4, <16 x i16> zeroinitializer, <16 x i16> %5
  store <16 x i16> %6, <16 x i16>* %2, align 2
  ret void
}

define void @test10(i8* nocapture %head) nounwind {
; SSE-LABEL: test10:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
; SSE-NEXT:    psubusb %xmm2, %xmm0
; SSE-NEXT:    psubusb %xmm2, %xmm1
; SSE-NEXT:    movdqu %xmm1, 16(%rdi)
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: test10:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovups (%rdi), %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpcmpgtb %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpcmpgtb %xmm0, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; AVX1-NEXT:    vxorps {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm0, %ymm1, %ymm0
; AVX1-NEXT:    vmovups %ymm0, (%rdi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test10:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovdqu (%rdi), %ymm0
; AVX2-NEXT:    vpsubusb {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <32 x i8>*
  %2 = load <32 x i8>, <32 x i8>* %1, align 1
  %3 = icmp slt <32 x i8> %2, zeroinitializer
  %4 = xor <32 x i8> %2, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %5 = select <32 x i1> %3, <32 x i8> %4, <32 x i8> zeroinitializer
  store <32 x i8> %5, <32 x i8>* %1, align 1
  ret void

}

define void @test11(i8* nocapture %head) nounwind {
; SSE-LABEL: test11:
; SSE:       ## BB#0: ## %vector.ph
; SSE-NEXT:    movdqu (%rdi), %xmm0
; SSE-NEXT:    movdqu 16(%rdi), %xmm1
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127]
; SSE-NEXT:    psubusb %xmm2, %xmm0
; SSE-NEXT:    psubusb %xmm2, %xmm1
; SSE-NEXT:    movdqu %xmm1, 16(%rdi)
; SSE-NEXT:    movdqu %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX1-LABEL: test11:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovups (%rdi), %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovaps {{.*#+}} xmm2 = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
; AVX1-NEXT:    vxorps %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254]
; AVX1-NEXT:    vpcmpgtb %xmm4, %xmm3, %xmm3
; AVX1-NEXT:    vxorps %xmm2, %xmm0, %xmm2
; AVX1-NEXT:    vpcmpgtb %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [129,129,129,129,129,129,129,129,129,129,129,129,129,129,129,129]
; AVX1-NEXT:    vpaddb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    vmovups %ymm0, (%rdi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test11:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovdqu (%rdi), %ymm0
; AVX2-NEXT:    vpsubusb {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <32 x i8>*
  %2 = load <32 x i8>, <32 x i8>* %1, align 1
  %3 = icmp ugt <32 x i8> %2, <i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126>
  %4 = add <32 x i8> %2, <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
  %5 = select <32 x i1> %3, <32 x i8> %4, <32 x i8> zeroinitializer
  store <32 x i8> %5, <32 x i8>* %1, align 1
  ret void
}

define void @test12(i8* nocapture %head, i8 zeroext %w) nounwind {
; SSE2-LABEL: test12:
; SSE2:       ## BB#0: ## %vector.ph
; SSE2-NEXT:    movd %esi, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,1,0,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    movdqu (%rdi), %xmm1
; SSE2-NEXT:    movdqu 16(%rdi), %xmm2
; SSE2-NEXT:    psubusb %xmm0, %xmm1
; SSE2-NEXT:    psubusb %xmm0, %xmm2
; SSE2-NEXT:    movdqu %xmm2, 16(%rdi)
; SSE2-NEXT:    movdqu %xmm1, (%rdi)
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: test12:
; SSSE3:       ## BB#0: ## %vector.ph
; SSSE3-NEXT:    movd %esi, %xmm0
; SSSE3-NEXT:    pxor %xmm1, %xmm1
; SSSE3-NEXT:    pshufb %xmm1, %xmm0
; SSSE3-NEXT:    movdqu (%rdi), %xmm1
; SSSE3-NEXT:    movdqu 16(%rdi), %xmm2
; SSSE3-NEXT:    psubusb %xmm0, %xmm1
; SSSE3-NEXT:    psubusb %xmm0, %xmm2
; SSSE3-NEXT:    movdqu %xmm2, 16(%rdi)
; SSSE3-NEXT:    movdqu %xmm1, (%rdi)
; SSSE3-NEXT:    retq
;
; AVX1-LABEL: test12:
; AVX1:       ## BB#0: ## %vector.ph
; AVX1-NEXT:    vmovups (%rdi), %ymm0
; AVX1-NEXT:    vmovd %esi, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpsubb %xmm1, %xmm2, %xmm3
; AVX1-NEXT:    vpsubb %xmm1, %xmm0, %xmm4
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm4, %ymm3
; AVX1-NEXT:    vpmaxub %xmm1, %xmm2, %xmm4
; AVX1-NEXT:    vpcmpeqb %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vpmaxub %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpcmpeqb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    vandps %ymm3, %ymm0, %ymm0
; AVX1-NEXT:    vmovups %ymm0, (%rdi)
; AVX1-NEXT:    vzeroupper
; AVX1-NEXT:    retq
;
; AVX2-LABEL: test12:
; AVX2:       ## BB#0: ## %vector.ph
; AVX2-NEXT:    vmovd %esi, %xmm0
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqu (%rdi), %ymm1
; AVX2-NEXT:    vpsubusb %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rdi)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
vector.ph:
  %0 = insertelement <32 x i8> undef, i8 %w, i32 0
  %broadcast15 = shufflevector <32 x i8> %0, <32 x i8> undef, <32 x i32> zeroinitializer
  %1 = getelementptr inbounds i8, i8* %head, i64 0
  %2 = bitcast i8* %1 to <32 x i8>*
  %3 = load <32 x i8>, <32 x i8>* %2, align 1
  %4 = icmp ult <32 x i8> %3, %broadcast15
  %5 = sub <32 x i8> %3, %broadcast15
  %6 = select <32 x i1> %4, <32 x i8> zeroinitializer, <32 x i8> %5
  store <32 x i8> %6, <32 x i8>* %2, align 1
  ret void
}
