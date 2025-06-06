; REQUIRES: asserts
; RUN: opt -mattr=+simd128 -passes=loop-vectorize -debug-only=loop-vectorize -disable-output -S < %s 2>&1 | FileCheck %s

target triple = "wasm32"

define hidden i32 @i32_mac_s8(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i32_mac_s8'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i8, ptr %arrayidx, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i8, ptr %arrayidx1, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv2 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 3 for VF 2 For instruction:   %0 = load i8, ptr %arrayidx, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 3 for VF 2 For instruction:   %1 = load i8, ptr %arrayidx1, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv2 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %0 = load i8, ptr %arrayidx, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %conv = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load i8, ptr %arrayidx1, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %conv2 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %mul = mul nsw i32 %conv2, %conv
; CHECK: LV: Selecting VF: 4.
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.09
  %0 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.09
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_s16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i32_mac_s16'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = sext i16 %0 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv2 = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv = sext i16 %0 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv2 = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv = sext i16 %0 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv2 = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %mul = mul nsw i32 %conv2, %conv
; CHECK: LV: Selecting VF: 4.
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_s16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i64_mac_s16'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = sext i16 %0 to i64
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv2 = sext i16 %1 to i64
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nsw i64 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %conv = sext i16 %0 to i64
; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %conv2 = sext i16 %1 to i64
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul nsw i64 %conv2, %conv
; CHECK: LV: Selecting VF: 2.
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i64
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i64
  %mul = mul nsw i64 %conv2, %conv
  %add = add nsw i64 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_s32(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i64_mac_s32'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i32, ptr %arrayidx, align 4
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i32, ptr %arrayidx1, align 4
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul i32 %1, %0
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %conv = sext i32 %mul to i64

; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %0 = load i32, ptr %arrayidx, align 4
; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load i32, ptr %arrayidx1, align 4
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul i32 %1, %0
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %conv = sext i32 %mul to i64
; CHECK: LV: Selecting VF: 2.
entry:
  %cmp6.not = icmp eq i32 %N, 0
  br i1 %cmp6.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.07 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %a, i32 %i.08
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %b, i32 %i.08
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul i32 %1, %0
  %conv = sext i32 %mul to i64
  %add = add i64 %res.07, %conv
  %inc = add nuw i32 %i.08, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_u8(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i32_mac_u8'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i8, ptr %arrayidx, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = zext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i8, ptr %arrayidx1, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv2 = zext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nuw nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 3 for VF 2 For instruction:   %0 = load i8, ptr %arrayidx, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv = zext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 3 for VF 2 For instruction:   %1 = load i8, ptr %arrayidx1, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv2 = zext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul nuw nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %0 = load i8, ptr %arrayidx, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %conv = zext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load i8, ptr %arrayidx1, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %conv2 = zext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %mul = mul nuw nsw i32 %conv2, %conv
; CHECK: LV: Selecting VF: 4.
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i32 %i.09
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i32 %i.09
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = zext i8 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i32 @i32_mac_u16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i32_mac_u16'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = zext i16 %0 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv2 = zext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nuw nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv = zext i16 %0 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv2 = zext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul nuw nsw i32 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv = zext i16 %0 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv2 = zext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %mul = mul nuw nsw i32 %conv2, %conv
; CHECK: LV: Selecting VF: 4.
entry:
  %cmp7.not = icmp eq i32 %N, 0
  br i1 %cmp7.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.09
  %0 = load i16, ptr %arrayidx, align 2
  %conv = zext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.09
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = zext i16 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add i32 %mul, %res.08
  %inc = add nuw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_u16(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i64_mac_u16'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = zext i16 %0 to i64
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv2 = zext i16 %1 to i64
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nuw nsw i64 %conv2, %conv

; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %0 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %conv = zext i16 %0 to i64
; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load i16, ptr %arrayidx1, align 2
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %conv2 = zext i16 %1 to i64
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul nuw nsw i64 %conv2, %conv
; CHECK: LV: Selecting VF: 2.
entry:
  %cmp8.not = icmp eq i32 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.09 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %a, i32 %i.010
  %0 = load i16, ptr %arrayidx, align 2
  %conv = zext i16 %0 to i64
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %b, i32 %i.010
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = zext i16 %1 to i64
  %mul = mul nuw nsw i64 %conv2, %conv
  %add = add i64 %mul, %res.09
  %inc = add nuw i32 %i.010, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define hidden i64 @i64_mac_u32(ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %b, i32 noundef %N) {
; CHECK-LABEL: 'i64_mac_u32'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i32, ptr %arrayidx, align 4
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i32, ptr %arrayidx1, align 4
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul i32 %1, %0
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %conv = zext i32 %mul to i64

; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %0 = load i32, ptr %arrayidx, align 4
; CHECK: LV: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load i32, ptr %arrayidx1, align 4
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %mul = mul i32 %1, %0
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %conv = zext i32 %mul to i64
; CHECK: LV: Selecting VF: 2.
entry:
  %cmp6.not = icmp eq i32 %N, 0
  br i1 %cmp6.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %for.body ]
  ret i64 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.07 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %a, i32 %i.08
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %b, i32 %i.08
  %1 = load i32, ptr %arrayidx1, align 4
  %mul = mul i32 %1, %0
  %conv = zext i32 %mul to i64
  %add = add i64 %res.07, %conv
  %inc = add nuw i32 %i.08, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
