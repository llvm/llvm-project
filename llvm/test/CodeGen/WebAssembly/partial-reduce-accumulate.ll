; RUN: opt -mattr=+simd128 -passes=loop-vectorize %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s
; RUN: opt -mattr=+simd128 -passes=loop-vectorize -vectorizer-maximize-bandwidth %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s --check-prefix=MAX-BANDWIDTH

target triple = "wasm32"

define hidden i32 @accumulate_add_u8_u8(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_add_u8_u8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_u
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_u
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add

entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.09 = phi i32 [ %add3, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.010
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.010
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = zext i8 %1 to i32
  %add = add i32 %result.09, %conv
  %add3 = add i32 %add, %conv2
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_add_s8_s8(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_add_s8_s8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.09 = phi i32 [ %add3, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.010
  %0 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.010
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %add = add i32 %result.09, %conv
  %add3 = add i32 %add, %conv2
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_add_s8_u8(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_add_s8_u8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_u
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.09 = phi i32 [ %add3, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.010
  %0 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.010
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = zext i8 %1 to i32
  %add = add i32 %result.09, %conv
  %add3 = add i32 %add, %conv2
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_add_s8_s16(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_add_s8_s16:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i32x4.load16x4_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extend_high_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.extend_high_i16x8_s
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.09 = phi i32 [ %add3, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.010
  %0 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.010
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %add = add i32 %result.09, %conv
  %add3 = add i32 %add, %conv2
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_shr_u8(ptr noundef readonly  %a, i32 noundef %N) {
; CHECK-LABEL: accumulate_shr_u8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i8x16.shr_u
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shr_u
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_u
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp4.not = icmp eq i32 %N, 0
  br i1 %cmp4.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.06
  %0 = load i8, ptr %arrayidx, align 1
  %1 = lshr i8 %0, 1
  %shr = zext nneg i8 %1 to i32
  %add = add i32 %result.05, %shr
  %inc = add nuw i32 %i.06, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_shr_s8(ptr noundef readonly  %a, i32 noundef %N) {
; CHECK-LABEL: accumulate_shr_s8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i8x16.shr_s
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shr_s
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp4.not = icmp eq i32 %N, 0
  br i1 %cmp4.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.06
  %0 = load i8, ptr %arrayidx, align 1
  %1 = ashr i8 %0, 1
  %shr = sext i8 %1 to i32
  %add = add nsw i32 %result.05, %shr
  %inc = add nuw i32 %i.06, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_max_u8_u8(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_max_u8_u8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: v128.load32_zero
; CHECK: i8x16.max_u
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.max_u
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_u
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp17.not = icmp eq i32 %N, 0
  br i1 %cmp17.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.019 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.018 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.019
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.019
  %1 = load i8, ptr %arrayidx1, align 1
  %. = tail call i8 @llvm.umax.i8(i8 %0, i8 %1)
  %cond = zext i8 %. to i32
  %add = add i32 %result.018, %cond
  %inc = add nuw i32 %i.019, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_min_s8_s8(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_min_s8_s8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: v128.load32_zero
; CHECK: i8x16.min_s
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.min_s
; MAX-BANDWIDTH: i16x8.extadd_pairwise_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp17.not = icmp eq i32 %N, 0
  br i1 %cmp17.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.019 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.018 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.019
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.019
  %1 = load i8, ptr %arrayidx1, align 1
  %. = tail call i8 @llvm.smin.i8(i8 %0, i8 %1)
  %cond = sext i8 %. to i32
  %add = add nsw i32 %result.018, %cond
  %inc = add nuw i32 %i.019, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_add_u16_u16(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_add_u16_u16:
; CHECK: loop
; CHECK: i32x4.load16x4_u
; CHECK: i32x4.add
; CHECK: i32x4.load16x4_u
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.09 = phi i32 [ %add3, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.010
  %0 = load i16, ptr %arrayidx, align 2
  %conv = zext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.010
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = zext i16 %1 to i32
  %add = add i32 %result.09, %conv
  %add3 = add i32 %add, %conv2
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_add_s16_s16(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_add_s16_s16:
; CHECK: loop
; CHECK: i32x4.load16x4_s
; CHECK: i32x4.add
; CHECK: i32x4.load16x4_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.09 = phi i32 [ %add3, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.010
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.010
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %add = add i32 %result.09, %conv
  %add3 = add i32 %add, %conv2
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_shr_u16(ptr noundef readonly  %a, i32 noundef %N) {
; CHECK-LABEL: accumulate_shr_u16:
; CHECK: loop
; CHECK: v128.load64_zero
; CHECK: i16x8.shr_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.shr_u
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_u
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp4.not = icmp eq i32 %N, 0
  br i1 %cmp4.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.06
  %0 = load i16, ptr %arrayidx, align 2
  %1 = lshr i16 %0, 1
  %shr = zext nneg i16 %1 to i32
  %add = add i32 %result.05, %shr
  %inc = add nuw i32 %i.06, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_shr_s16(ptr noundef readonly  %a, i32 noundef %N) {
; CHECK-LABEL: accumulate_shr_s16:
; CHECK: loop
; CHECK: v128.load64_zero
; CHECK: i16x8.shr_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.shr_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
entry:
  %cmp4.not = icmp eq i32 %N, 0
  br i1 %cmp4.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.06
  %0 = load i16, ptr %arrayidx, align 2
  %1 = ashr i16 %0, 1
  %shr = sext i16 %1 to i32
  %add = add nsw i32 %result.05, %shr
  %inc = add nuw i32 %i.06, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_sub_s8_s8(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_sub_s8_s8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.add
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: i32x4.sub

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.sub
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.sub
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.sub
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.09
  %0 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.09
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %sub = add i32 %result.08, %conv
  %add = sub i32 %sub, %conv2
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @accumulate_sub_s16_s16(ptr noundef readonly  %a, ptr noundef readonly  %b, i32 noundef %N) {
; CHECK-LABEL: accumulate_sub_s16_s16:
; CHECK: loop
; CHECK: i32x4.load16x4_s
; CHECK: i32x4.add
; CHECK: i32x4.load16x4_s
; CHECK: i32x4.sub

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extend_high_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extend_high_i16x8_s
; MAX-BANDWIDTH: i32x4.sub
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.sub
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %result.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %sub = add i32 %result.08, %conv
  %add = sub i32 %sub, %conv2
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare i8 @llvm.umax.i8(i8, i8)

declare i8 @llvm.smin.i8(i8, i8)
