; RUN: llc -mcpu=core2 < %s | FileCheck %s -check-prefix=SSSE3
; RUN: llc -mcpu=corei7-avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -mcpu=core-avx2 < %s | FileCheck %s -check-prefix=AVX2

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @test1(i16* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <8 x i16>*
  %2 = load <8 x i16>, <8 x i16>* %1, align 2
  %3 = icmp slt <8 x i16> %2, zeroinitializer
  %4 = xor <8 x i16> %2, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %5 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> zeroinitializer
  store <8 x i16> %5, <8 x i16>* %1, align 2
  ret void

; SSSE3: @test1
; SSSE3:      # BB#0:
; SSSE3-NEXT: movdqu (%rdi), %xmm0
; SSSE3-NEXT: psubusw LCPI0_0(%rip), %xmm0
; SSSE3-NEXT: movdqu %xmm0, (%rdi)
; SSSE3-NEXT: retq

; AVX1: @test1
; AVX1:      # BB#0:
; AVX1-NEXT: vmovdqu (%rdi), %xmm0
; AVX1-NEXT: vpsubusw LCPI0_0(%rip), %xmm0, %xmm0
; AVX1-NEXT: vmovdqu %xmm0, (%rdi)
; AVX1-NEXT: retq

; AVX2: @test1
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %xmm0
; AVX2-NEXT: vpsubusw LCPI0_0(%rip), %xmm0, %xmm0
; AVX2-NEXT: vmovdqu %xmm0, (%rdi)
; AVX2-NEXT: retq
}

define void @test2(i16* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <8 x i16>*
  %2 = load <8 x i16>, <8 x i16>* %1, align 2
  %3 = icmp ugt <8 x i16> %2, <i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766>
  %4 = add <8 x i16> %2, <i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767>
  %5 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> zeroinitializer
  store <8 x i16> %5, <8 x i16>* %1, align 2
  ret void

; SSSE3: @test2
; SSSE3:      # BB#0:
; SSSE3-NEXT: movdqu (%rdi), %xmm0
; SSSE3-NEXT: psubusw LCPI1_0(%rip), %xmm0
; SSSE3-NEXT: movdqu %xmm0, (%rdi)
; SSSE3-NEXT: retq

; AVX1: @test2
; AVX1:      # BB#0:
; AVX1-NEXT: vmovdqu (%rdi), %xmm0
; AVX1-NEXT: vpsubusw LCPI1_0(%rip), %xmm0, %xmm0
; AVX1-NEXT: vmovdqu %xmm0, (%rdi)
; AVX1-NEXT: retq

; AVX2: @test2
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %xmm0
; AVX2-NEXT: vpsubusw LCPI1_0(%rip), %xmm0, %xmm0
; AVX2-NEXT: vmovdqu %xmm0, (%rdi)
; AVX2-NEXT: retq
}

define void @test3(i16* nocapture %head, i16 zeroext %w) nounwind {
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

; SSSE3: @test3
; SSSE3:      # BB#0:
; SSSE3-NEXT: movd %esi, %xmm0
; SSSE3-NEXT: pshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; SSSE3-NEXT: movdqu (%rdi), %xmm1
; SSSE3-NEXT: psubusw %xmm0, %xmm1
; SSSE3-NEXT: movdqu %xmm1, (%rdi)
; SSSE3-NEXT: retq

; AVX1: @test3
; AVX1:      # BB#0:
; AVX1-NEXT: vmovd %esi, %xmm0
; AVX1-NEXT: vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT: vmovdqu (%rdi), %xmm1
; AVX1-NEXT: vpsubusw %xmm0, %xmm1, %xmm0
; AVX1-NEXT: vmovdqu %xmm0, (%rdi)
; AVX1-NEXT: retq

; AVX2: @test3
; AVX2:      # BB#0:
; AVX2-NEXT: vmovd %esi, %xmm0
; AVX2-NEXT: vpbroadcastw %xmm0, %xmm0
; AVX2-NEXT: vmovdqu (%rdi), %xmm1
; AVX2-NEXT: vpsubusw %xmm0, %xmm1, %xmm0
; AVX2-NEXT: vmovdqu %xmm0, (%rdi)
; AVX2-NEXT: retq
}

define void @test4(i8* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <16 x i8>*
  %2 = load <16 x i8>, <16 x i8>* %1, align 1
  %3 = icmp slt <16 x i8> %2, zeroinitializer
  %4 = xor <16 x i8> %2, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %5 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> zeroinitializer
  store <16 x i8> %5, <16 x i8>* %1, align 1
  ret void

; SSSE3: @test4
; SSSE3:      # BB#0:
; SSSE3-NEXT: movdqu (%rdi), %xmm0
; SSSE3-NEXT: psubusb LCPI3_0(%rip), %xmm0
; SSSE3-NEXT: movdqu %xmm0, (%rdi)
; SSSE3-NEXT: retq

; AVX1: @test4
; AVX1:      # BB#0:
; AVX1-NEXT: vmovdqu (%rdi), %xmm0
; AVX1-NEXT: vpsubusb LCPI3_0(%rip), %xmm0, %xmm0
; AVX1-NEXT: vmovdqu %xmm0, (%rdi)
; AVX1-NEXT: retq

; AVX2: @test4
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %xmm0
; AVX2-NEXT: vpsubusb LCPI3_0(%rip), %xmm0, %xmm0
; AVX2-NEXT: vmovdqu %xmm0, (%rdi)
; AVX2-NEXT: retq
}

define void @test5(i8* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <16 x i8>*
  %2 = load <16 x i8>, <16 x i8>* %1, align 1
  %3 = icmp ugt <16 x i8> %2, <i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126>
  %4 = add <16 x i8> %2, <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
  %5 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> zeroinitializer
  store <16 x i8> %5, <16 x i8>* %1, align 1
  ret void

; SSSE3: @test5
; SSSE3:      # BB#0:
; SSSE3-NEXT: movdqu (%rdi), %xmm0
; SSSE3-NEXT: psubusb LCPI4_0(%rip), %xmm0
; SSSE3-NEXT: movdqu %xmm0, (%rdi)
; SSSE3-NEXT: retq

; AVX1: @test5
; AVX1:      # BB#0:
; AVX1-NEXT: vmovdqu (%rdi), %xmm0
; AVX1-NEXT: vpsubusb LCPI4_0(%rip), %xmm0
; AVX1-NEXT: vmovdqu %xmm0, (%rdi)
; AVX1-NEXT: retq

; AVX2: @test5
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %xmm0
; AVX2-NEXT: vpsubusb LCPI4_0(%rip), %xmm0
; AVX2-NEXT: vmovdqu %xmm0, (%rdi)
; AVX2-NEXT: retq
}

define void @test6(i8* nocapture %head, i8 zeroext %w) nounwind {
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

; SSSE3: @test6
; SSSE3:      # BB#0:
; SSSE3-NEXT: movd %esi, %xmm0
; SSSE3-NEXT: pxor %xmm1, %xmm1
; SSSE3-NEXT: pshufb %xmm1, %xmm0
; SSSE3-NEXT: movdqu (%rdi), %xmm1
; SSSE3-NEXT: psubusb %xmm0, %xmm1
; SSSE3-NEXT: movdqu %xmm1, (%rdi)
; SSSE3-NEXT: retq

; AVX1: @test6
; AVX1:      # BB#0:
; AVX1-NEXT: vmovd %esi, %xmm0
; AVX1-NEXT: vpxor %xmm1, %xmm1
; AVX1-NEXT: vpshufb %xmm1, %xmm0
; AVX1-NEXT: vmovdqu (%rdi), %xmm1
; AVX1-NEXT: vpsubusb %xmm0, %xmm1, %xmm0
; AVX1-NEXT: vmovdqu %xmm0, (%rdi)
; AVX1-NEXT: retq

; AVX2: @test6
; AVX2:      # BB#0:
; AVX2-NEXT: vmovd %esi, %xmm0
; AVX2-NEXT: vpbroadcastb %xmm0, %xmm0
; AVX2-NEXT: vmovdqu (%rdi), %xmm1
; AVX2-NEXT: vpsubusb %xmm0, %xmm1, %xmm0
; AVX2-NEXT: vmovdqu %xmm0, (%rdi)
; AVX2-NEXT: retq
}

define void @test7(i16* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <16 x i16>*
  %2 = load <16 x i16>, <16 x i16>* %1, align 2
  %3 = icmp slt <16 x i16> %2, zeroinitializer
  %4 = xor <16 x i16> %2, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %5 = select <16 x i1> %3, <16 x i16> %4, <16 x i16> zeroinitializer
  store <16 x i16> %5, <16 x i16>* %1, align 2
  ret void

; AVX2: @test7
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %ymm0
; AVX2-NEXT: vpsubusw LCPI6_0(%rip), %ymm0, %ymm0
; AVX2-NEXT: vmovdqu %ymm0, (%rdi)
; AVX2-NEXT: vzeroupper
; AVX2-NEXT: retq
}

define void @test8(i16* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i16, i16* %head, i64 0
  %1 = bitcast i16* %0 to <16 x i16>*
  %2 = load <16 x i16>, <16 x i16>* %1, align 2
  %3 = icmp ugt <16 x i16> %2, <i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766, i16 32766>
  %4 = add <16 x i16> %2, <i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767, i16 -32767>
  %5 = select <16 x i1> %3, <16 x i16> %4, <16 x i16> zeroinitializer
  store <16 x i16> %5, <16 x i16>* %1, align 2
  ret void

; AVX2: @test8
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %ymm0
; AVX2-NEXT: vpsubusw LCPI7_0(%rip), %ymm0, %ymm0
; AVX2-NEXT: vmovdqu %ymm0, (%rdi)
; AVX2-NEXT: vzeroupper
; AVX2-NEXT: retq
}

define void @test9(i16* nocapture %head, i16 zeroext %w) nounwind {
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

; AVX2: @test9
; AVX2:      # BB#0:
; AVX2-NEXT: vmovd %esi, %xmm0
; AVX2-NEXT: vpbroadcastw %xmm0, %ymm0
; AVX2-NEXT: vmovdqu (%rdi), %ymm1
; AVX2-NEXT: vpsubusw %ymm0, %ymm1, %ymm0
; AVX2-NEXT: vmovdqu %ymm0, (%rdi)
; AVX2-NEXT: vzeroupper
; AVX2-NEXT: retq
}

define void @test10(i8* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <32 x i8>*
  %2 = load <32 x i8>, <32 x i8>* %1, align 1
  %3 = icmp slt <32 x i8> %2, zeroinitializer
  %4 = xor <32 x i8> %2, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %5 = select <32 x i1> %3, <32 x i8> %4, <32 x i8> zeroinitializer
  store <32 x i8> %5, <32 x i8>* %1, align 1
  ret void

; AVX2: @test10
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %ymm0
; AVX2-NEXT: vpsubusb LCPI9_0(%rip), %ymm0, %ymm0
; AVX2-NEXT: vmovdqu %ymm0, (%rdi)
; AVX2-NEXT: vzeroupper
; AVX2-NEXT: retq
}

define void @test11(i8* nocapture %head) nounwind {
vector.ph:
  %0 = getelementptr inbounds i8, i8* %head, i64 0
  %1 = bitcast i8* %0 to <32 x i8>*
  %2 = load <32 x i8>, <32 x i8>* %1, align 1
  %3 = icmp ugt <32 x i8> %2, <i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126, i8 126>
  %4 = add <32 x i8> %2, <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
  %5 = select <32 x i1> %3, <32 x i8> %4, <32 x i8> zeroinitializer
  store <32 x i8> %5, <32 x i8>* %1, align 1
  ret void

; AVX2: @test11
; AVX2:      # BB#0:
; AVX2-NEXT: vmovdqu (%rdi), %ymm0
; AVX2-NEXT: vpsubusb LCPI10_0(%rip), %ymm0, %ymm0
; AVX2-NEXT: vmovdqu %ymm0, (%rdi)
; AVX2-NEXT: vzeroupper
; AVX2-NEXT: retq
}

define void @test12(i8* nocapture %head, i8 zeroext %w) nounwind {
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

; AVX2: @test12
; AVX2:      # BB#0:
; AVX2-NEXT: vmovd %esi, %xmm0
; AVX2-NEXT: vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT: vmovdqu (%rdi), %ymm1
; AVX2-NEXT: vpsubusb %ymm0, %ymm1, %ymm0
; AVX2-NEXT: vmovdqu %ymm0, (%rdi)
; AVX2-NEXT: vzeroupper
; AVX2-NEXT: retq
}
