; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mattr=+avx2     -S | FileCheck %s --check-prefix=AVX2
; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mattr=+avx512bw -S | FileCheck %s --check-prefix=AVX512

; Wider-VF i64-accumulator positive shapes for tryByteSumReplacement.
; These exercise the PerSplitTy = <IntrinsicNumElts/8 x i64> branch where the
; matcher must NOT bitcast the psadbw result back to i32.
;
; VF=32 i64 uses the AVX2 256-bit lane (one avx2.psad.bw call). On
; +avx512bw the dispatch falls into the same AVX2 path because NumElts<64.
;
; VF=64 i64 uses the AVX-512BW 512-bit lane (one avx512.psad.bw.512 call).

@a = global [1024 x i8] zeroinitializer, align 16

; AVX2-LABEL: @byte_sum_v32_i64(
; AVX2: call <4 x i64> @llvm.x86.avx2.psad.bw(
; AVX512-LABEL: @byte_sum_v32_i64(
; AVX512: call <4 x i64> @llvm.x86.avx2.psad.bw(
define i64 @byte_sum_v32_i64() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <32 x i64> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <32 x i8>, ptr %p, align 16
  %z = zext <32 x i8> %wide.load to <32 x i64>
  %add = add nsw <32 x i64> %z, %vec.phi
  %index.next = add i64 %index, 32
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i64 @llvm.vector.reduce.add.v32i64(<32 x i64> %add)
  ret i64 %ext
}

; AVX512-LABEL: @byte_sum_v64_i64(
; AVX512: call <8 x i64> @llvm.x86.avx512.psad.bw.512(
define i64 @byte_sum_v64_i64() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <64 x i64> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <64 x i8>, ptr %p, align 16
  %z = zext <64 x i8> %wide.load to <64 x i64>
  %add = add nsw <64 x i64> %z, %vec.phi
  %index.next = add i64 %index, 64
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i64 @llvm.vector.reduce.add.v64i64(<64 x i64> %add)
  ret i64 %ext
}
