; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f | FileCheck %s --check-prefix=AVX512
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512bw | FileCheck %s --check-prefix=AVX512

;SSE2-label: @_Z10test_shortPsS_i
;SSE2:        movdqu
;SSE2-NEXT:   movdqu
;SSE2-NEXT:   pmaddwd
;SSE2-NEXT:   paddd

;AVX2-label: @_Z10test_shortPsS_i
;AVX2:        vmovdqu
;AVX2-NEXT:   vpmaddwd
;AVX2-NEXT:   vinserti128
;AVX2-NEXT:   vpaddd

;AVX512-label: @_Z10test_shortPsS_i
;AVX512:        vmovdqu
;AVX512-NEXT:   vpmaddwd
;AVX512-NEXT:   vinserti128
;AVX512-NEXT:   vpaddd

define i32 @_Z10test_shortPsS_i(i16* nocapture readonly, i16* nocapture readonly, i32) local_unnamed_addr #0 {
entry:
  %3 = zext i32 %2 to i64
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %vec.phi = phi <8 x i32> [ %11, %vector.body ], [ zeroinitializer, %entry ]
  %4 = getelementptr inbounds i16, i16* %0, i64 %index
  %5 = bitcast i16* %4 to <8 x i16>*
  %wide.load = load <8 x i16>, <8 x i16>* %5, align 2
  %6 = sext <8 x i16> %wide.load to <8 x i32>
  %7 = getelementptr inbounds i16, i16* %1, i64 %index
  %8 = bitcast i16* %7 to <8 x i16>*
  %wide.load14 = load <8 x i16>, <8 x i16>* %8, align 2
  %9 = sext <8 x i16> %wide.load14 to <8 x i32>
  %10 = mul nsw <8 x i32> %9, %6
  %11 = add nsw <8 x i32> %10, %vec.phi
  %index.next = add i64 %index, 8
  %12 = icmp eq i64 %index.next, %3
  br i1 %12, label %middle.block, label %vector.body

middle.block:
  %rdx.shuf = shufflevector <8 x i32> %11, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i32> %11, %rdx.shuf
  %rdx.shuf15 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx16 = add <8 x i32> %bin.rdx, %rdx.shuf15
  %rdx.shuf17 = shufflevector <8 x i32> %bin.rdx16, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx18 = add <8 x i32> %bin.rdx16, %rdx.shuf17
  %13 = extractelement <8 x i32> %bin.rdx18, i32 0
  ret i32 %13
}

;AVX2-label: @_Z9test_charPcS_i
;AVX2:       vpmovsxbw
;AVX2-NEXT:  vpmovsxbw
;AVX2-NEXT:  vpmaddwd
;AVX2-NEXT:  vpaddd

;AVX512-label: @_Z9test_charPcS_i
;AVX512:       vpmovsxbw
;AVX512-NEXT:  vpmovsxbw
;AVX512-NEXT:  vpmaddwd
;AVX512-NEXT:  vinserti64x4
;AVX512-NEXT:  vpaddd

define i32 @_Z9test_charPcS_i(i8* nocapture readonly, i8* nocapture readonly, i32) local_unnamed_addr #0 {
entry:
  %3 = zext i32 %2 to i64
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %vec.phi = phi <16 x i32> [ %11, %vector.body ], [ zeroinitializer, %entry ]
  %4 = getelementptr inbounds i8, i8* %0, i64 %index
  %5 = bitcast i8* %4 to <16 x i8>*
  %wide.load = load <16 x i8>, <16 x i8>* %5, align 1
  %6 = sext <16 x i8> %wide.load to <16 x i32>
  %7 = getelementptr inbounds i8, i8* %1, i64 %index
  %8 = bitcast i8* %7 to <16 x i8>*
  %wide.load14 = load <16 x i8>, <16 x i8>* %8, align 1
  %9 = sext <16 x i8> %wide.load14 to <16 x i32>
  %10 = mul nsw <16 x i32> %9, %6
  %11 = add nsw <16 x i32> %10, %vec.phi
  %index.next = add i64 %index, 16
  %12 = icmp eq i64 %index.next, %3
  br i1 %12, label %middle.block, label %vector.body

middle.block:
  %rdx.shuf = shufflevector <16 x i32> %11, <16 x i32> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <16 x i32> %11, %rdx.shuf
  %rdx.shuf15 = shufflevector <16 x i32> %bin.rdx, <16 x i32> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx16 = add <16 x i32> %bin.rdx, %rdx.shuf15
  %rdx.shuf17 = shufflevector <16 x i32> %bin.rdx16, <16 x i32> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx18 = add <16 x i32> %bin.rdx16, %rdx.shuf17
  %rdx.shuf19 = shufflevector <16 x i32> %bin.rdx18, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx20 = add <16 x i32> %bin.rdx18, %rdx.shuf19
  %13 = extractelement <16 x i32> %bin.rdx20, i32 0
  ret i32 %13
}
