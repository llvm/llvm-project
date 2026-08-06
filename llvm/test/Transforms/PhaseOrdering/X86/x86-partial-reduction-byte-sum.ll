; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mcpu=x86-64    -S | FileCheck %s --check-prefixes=CHECK,SSE
; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mcpu=x86-64-v2 -S | FileCheck %s --check-prefixes=CHECK,SSE
; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mcpu=x86-64-v3 -S | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt < %s -passes='expand-reductions,x86-partial-reduction' -mtriple=x86_64-unknown-unknown -mcpu=x86-64-v4 -S | FileCheck %s --check-prefixes=CHECK,AVX512

; Test X86PartialReduction::tryByteSumReplacement: positive and negative shapes.

@a = global [1024 x i8] zeroinitializer, align 16

;; Positive cases -----------------------------------------------------------

; CHECK-LABEL: @byte_sum_v16_i32(
; CHECK: call <2 x i64> @llvm.x86.sse2.psad.bw(
define i32 @byte_sum_v16_i32() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <16 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <16 x i8>, ptr %p, align 16
  %z = zext <16 x i8> %wide.load to <16 x i32>
  %add = add nsw <16 x i32> %z, %vec.phi
  %index.next = add i64 %index, 16
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %add)
  ret i32 %ext
}

; SSE-LABEL: @byte_sum_v32_i64(
; SSE: call <2 x i64> @llvm.x86.sse2.psad.bw(
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

; SSE-LABEL: @byte_sum_v64_i64(
; SSE: call <2 x i64> @llvm.x86.sse2.psad.bw(
; AVX2-LABEL: @byte_sum_v64_i64(
; AVX2: call <4 x i64> @llvm.x86.avx2.psad.bw(
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

;; Negative cases -----------------------------------------------------------

; CHECK-LABEL: @byte_sum_v8_i32(
; CHECK-NOT: psad.bw
; CHECK: ret i32
define i32 @byte_sum_v8_i32() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <8 x i8>, ptr %p, align 8
  %z = zext <8 x i8> %wide.load to <8 x i32>
  %add = add nsw <8 x i32> %z, %vec.phi
  %index.next = add i64 %index, 8
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %add)
  ret i32 %ext
}

; CHECK-LABEL: @byte_sum_v24_i32(
; CHECK-NOT: psad.bw
; CHECK: ret i32
define i32 @byte_sum_v24_i32() nounwind {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <24 x i32> [ zeroinitializer, %entry ], [ %add, %vector.body ]
  %p = getelementptr inbounds [1024 x i8], ptr @a, i64 0, i64 %index
  %wide.load = load <24 x i8>, ptr %p, align 8
  %z = zext <24 x i8> %wide.load to <24 x i32>
  %add = add nsw <24 x i32> %z, %vec.phi
  %index.next = add i64 %index, 24
  %cmp = icmp eq i64 %index.next, 1024
  br i1 %cmp, label %middle.block, label %vector.body

middle.block:
  %ext = call i32 @llvm.vector.reduce.add.v24i32(<24 x i32> %add)
  ret i32 %ext
}
