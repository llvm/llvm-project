; RUN: opt -mattr=+simd128 -passes=loop-vectorize %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s
; RUN: opt -mattr=+simd128 -passes=loop-vectorize -vectorizer-maximize-bandwidth %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s --check-prefix=MAX-BANDWIDTH

target triple = "wasm32"

define hidden i32 @i32_mac_s8(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i32_mac_s8:
; CHECK:    v128.load32_zero 0:p2align=0
; CHECK:    i16x8.extend_low_i8x16_s
; CHECK:    v128.load32_zero 0:p2align=0
; CHECK:    i16x8.extend_low_i8x16_s
; CHECK:    i32x4.extmul_low_i16x8_s
; CHECK:    i32x4.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extend_low_i8x16_s
; MAX-BANDWIDTH: i32x4.dot_i16x8_s
; MAX-BANDWIDTH: i16x8.extend_high_i8x16_s
; MAX-BANDWIDTH: i16x8.extend_high_i8x16_s
; MAX-BANDWIDTH: i32x4.dot_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add

entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i32 %i.09
  %0 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8, ptr %b, i32 %i.09
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_s16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i32_mac_s16:
; CHECK:    i32x4.load16x4_s 0:p2align=1
; CHECK:    i32x4.load16x4_s 0:p2align=1
; CHECK:    i32x4.mul
; CHECK:    i32x4.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.dot_i16x8_s

entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_s16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i64_mac_s16:
; CHECK:    v128.load32_zero 0:p2align=1
; CHECK:    i32x4.extend_low_i16x8_s
; CHECK:    v128.load32_zero 0:p2align=1
; CHECK:    i32x4.extend_low_i16x8_s
; CHECK:    i64x2.extmul_low_i32x4_s
; CHECK:    i64x2.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_s
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_s
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_s
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_s
; MAX-BANDWIDTH: i64x2.add

entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i64
  %arrayidx1 = getelementptr inbounds i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i64
  %mul = mul nsw i64 %conv2, %conv
  %add = add nsw i64 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_s32(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i64_mac_s32:
; CHECK:    v128.load64_zero 0:p2align=2
; CHECK:    v128.load64_zero 0:p2align=2
; CHECK:    i32x4.mul
; CHECK:    i64x2.extend_low_i32x4_s
; CHECK:    i64x2.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.mul
; MAX-BANDWIDTH: i64x2.extend_low_i32x4_s
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
; MAX-BANDWIDTH: i64x2.extend_low_i32x4_s
; MAX-BANDWIDTH: i64x2.add

entry:
  %cmp6.not = icmp eq i32 %N, 0
  br i1 %cmp6.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.07 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.08
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i32 %i.08
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul i32 %1, %0
  %conv = sext i32 %mul to i64
  %add = add i64 %res.07, %conv
  %inc = add nuw i32 %i.08, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_u8(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i32_mac_u8:
; CHECK:    v128.load32_zero 0:p2align=0
; CHECK:    i16x8.extend_low_i8x16_u
; CHECK:    v128.load32_zero 0:p2align=0
; CHECK:    i16x8.extend_low_i8x16_u
; CHECK:    i32x4.extmul_low_i16x8_u
; CHECK:    i32x4.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_u
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i32x4.extend_high_i16x8_u
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_u
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i32x4.extend_high_i16x8_u
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add

entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i32 %i.09
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8, ptr %b, i32 %i.09
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = zext i8 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_u16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i32_mac_u16:
; CHECK:    i32x4.load16x4_u 0:p2align=1
; CHECK:    i32x4.load16x4_u 0:p2align=1
; CHECK:    i32x4.mul
; CHECK:    i32x4.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extmul_low_i16x8_u
; MAX-BANDWIDTH: i32x4.extmul_high_i16x8_u
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add

entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = zext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = zext i16 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_u16_s16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i32_mac_u16_s16:
; CHECK:    i32x4.load16x4_s 0:p2align=1
; CHECK:    i32x4.load16x4_u 0:p2align=1
; CHECK:    i32x4.mul
; CHECK:    i32x4.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i32x4.mul
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_s
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i32x4.mul
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add

entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = zext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_u16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i64_mac_u16:
; CHECK:    v128.load32_zero 0:p2align=1
; CHECK:    i32x4.extend_low_i16x8_u
; CHECK:    v128.load32_zero 0:p2align=1
; CHECK:    i32x4.extend_low_i16x8_u
; CHECK:    i64x2.extmul_low_i32x4_u
; CHECK:    i64x2.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_u
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_u
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 6, 7, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_u
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i32x4.extend_low_i16x8_u
; MAX-BANDWIDTH: i64x2.extmul_low_i32x4_u
; MAX-BANDWIDTH: i64x2.add

entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.09 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %a, i32 %i.010
  %0 = load i16, ptr %arrayidx, align 2
  %conv = zext i16 %0 to i64
  %arrayidx1 = getelementptr inbounds i16, ptr %b, i32 %i.010
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = zext i16 %1 to i64
  %mul = mul nuw nsw i64 %conv2, %conv
  %add = add i64 %mul, %res.09
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_u32(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: i64_mac_u32:
; CHECK:    v128.load64_zero 0:p2align=2
; CHECK:    v128.load64_zero 0:p2align=2
; CHECK:    i32x4.mul
; CHECK:    i64x2.extend_low_i32x4_u
; CHECK:    i64x2.add

; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.mul
; MAX-BANDWIDTH: i64x2.extend_low_i32x4_u
; MAX-BANDWIDTH: i64x2.add
; MAX-BANDWIDTH: i8x16.shuffle	8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
; MAX-BANDWIDTH: i64x2.extend_low_i32x4_u
; MAX-BANDWIDTH: i64x2.add

entry:
  %cmp6.not = icmp eq i32 %N, 0
  br i1 %cmp6.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.07 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i.08
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i32 %i.08
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul i32 %1, %0
  %conv = zext i32 %mul to i64
  %add = add i64 %res.07, %conv
  %inc = add nuw i32 %i.08, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
