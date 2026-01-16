; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s

; Test how shufflevectors are re-vectorised using HVLA.

define void @unzip_even_same_src(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @unzip_even_same_src(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    call <vscale x 8 x i16> @llvm.vector.segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> poison
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 0, i32 2, i32 4, i32 6>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @unzip_odd(ptr noalias nocapture noundef writeonly %a, ptr readonly %b, ptr readonly %c) {
; CHECK-LABEL: define void @unzip_odd(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_LOAD2:%.*]] = load <vscale x 8 x i16>
; CHECK:    call <vscale x 8 x i16> @llvm.vector.segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> [[WIDE_LOAD2]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %arrayidx1 = getelementptr inbounds <8 x i16>, ptr %c, i64 %indvars.iv
  %1 = load <8 x i16>, ptr %arrayidx1, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> %1, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @trn1_single_src(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @trn1_single_src(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    call <vscale x 8 x i16> @llvm.vector.segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> poison
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @duplicate_element(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @duplicate_element(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    call <vscale x 8 x i16> @llvm.vector.segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> poison
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @duplicate_element_64bit(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @duplicate_element_64bit(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i16>
; CHECK:    call <vscale x 4 x i16> @llvm.vector.segmented.shuffle.nxv4i16.v4i32(<vscale x 4 x i16> [[WIDE_LOAD]], <vscale x 4 x i16> poison, <4 x i32> splat (i32 3))
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i16> %0, <4 x i16> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Cannot re-vectorise "duplicate element" shuffles that change size.
define void @duplicate_element_64bit_to_128bit(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @duplicate_element_64bit_to_128bit(
; CHECK-NOT:  vector.body:
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i16> %0, <4 x i16> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; For uniform shuffles, the resulting code is found in the preheader.
define void @duplicate_element_uniform(ptr noalias nocapture noundef writeonly %a, <8 x i16> %0) {
; CHECK-LABEL: define void @duplicate_element_uniform(
; CHECK:  vector.ph:
; CHECK:    [[BROADCAST:%.*]] = call <vscale x 8 x i16> @llvm.vector.broadcast.nxv8i16.v8i16
; CHECK:    call <vscale x 8 x i16> @llvm.vector.segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[BROADCAST]], <vscale x 8 x i16> poison, <8 x i32> splat (i32 3))
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @concat_shuffle(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @concat_shuffle(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i16>
; CHECK:    [[INT_SEGS:%.*]] = call <vscale x 8 x i16> @llvm.vector.interleave.segments2.nxv8i16(<vscale x 4 x i16> [[WIDE_LOAD]], <vscale x 4 x i16> splat (i16 1))
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i16> %0, <4 x i16> splat (i16 1), <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @concat_shuffle_undef(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @concat_shuffle_undef(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i16>
; CHECK:    [[INT_SEGS:%.*]] = call <vscale x 8 x i16> @llvm.vector.interleave.segments2.nxv8i16(<vscale x 4 x i16> [[WIDE_LOAD]], <vscale x 4 x i16> poison)
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i16> %0, <4 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @concat_shuffle_self(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @concat_shuffle_self(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i16>
; CHECK:    [[INT_SEGS:%.*]] = call <vscale x 8 x i16> @llvm.vector.interleave.segments2.nxv8i16(<vscale x 4 x i16> [[WIDE_LOAD]], <vscale x 4 x i16> [[WIDE_LOAD]])
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i16> %0, <4 x i16> poison, <8 x i32> <i32 0, i32 poison, i32 2, i32 3, i32 poison, i32 1, i32 2, i32 3>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @extract_shuffle_idx0(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @extract_shuffle_idx0(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[DEINT:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave.segments2.nxv8i16(<vscale x 8 x i16> [[WIDE_LOAD]])
; CHECK:    extractvalue { <vscale x 4 x i16>, <vscale x 4 x i16> } [[DEINT]], 0
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @extract_shuffle_idx1(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @extract_shuffle_idx1(
; CHECK:  vector.body:
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[DEINT:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave.segments2.nxv8i16(<vscale x 8 x i16> [[WIDE_LOAD]])
; CHECK:    extractvalue { <vscale x 4 x i16>, <vscale x 4 x i16> } [[DEINT]], 1
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Cannot re-vectorise "unaligned" extracts
define void @extract_shuffle_invalid_idx(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @extract_shuffle_invalid_idx(
; CHECK-NOT:  vector.body:
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Cannot re-vectorise "quarter" extracts
define void @extract_shuffle_factor4(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @extract_shuffle_factor4(
; CHECK-NOT:  vector.body:
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <2 x i32> <i32 0, i32 1>
  %arrayidx2 = getelementptr inbounds <2 x i16>, ptr %a, i64 %indvars.iv
  store <2 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Cannot re-vectorise extracts from 64-bit NEON
define void @extract_shuffle_from_64bits(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @extract_shuffle_from_64bits(
; CHECK-NOT:  vector.body:
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i16> %0, <4 x i16> poison, <2 x i32> <i32 0, i32 1>
  %arrayidx2 = getelementptr inbounds <2 x i16>, ptr %a, i64 %indvars.iv
  store <2 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @arbitrary_shuffle(ptr noalias nocapture noundef writeonly %a, ptr readonly %b) {
; CHECK-LABEL: define void @arbitrary_shuffle(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    call <vscale x 8 x i16> @llvm.vector.segmented.shuffle.nxv8i16.v8i32(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> poison
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> <i32 3, i32 2, i32 3, i32 6, i32 3, i32 1, i32 3, i32 0>
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
