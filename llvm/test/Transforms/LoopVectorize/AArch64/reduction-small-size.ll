; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -dce -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: @reduction_i8
;
; char reduction_i8(char *a, char *b, int n) {
;   char sum = 0;
;   for (int i = 0; i < n; ++i)
;     sum += (a[i] + b[i]);
;   return sum;
; }
;
; CHECK: vector.body:
; CHECK:   phi <16 x i8>
; CHECK:   load <16 x i8>
; CHECK:   load <16 x i8>
; CHECK:   add <16 x i8>
; CHECK:   add <16 x i8>
;
; CHECK: middle.block:
; CHECK:   [[Rdx:%[a-zA-Z0-9.]+]] = call i8 @llvm.experimental.vector.reduce.add.i8.v16i8(<16 x i8>
; CHECK:   zext i8 [[Rdx]] to i32
;
define i8 @reduction_i8(i8* nocapture readonly %a, i8* nocapture readonly %b, i32 %n) {
entry:
  %cmp.12 = icmp sgt i32 %n, 0
  br i1 %cmp.12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  %add5.lcssa = phi i32 [ %add5, %for.body ]
  %conv6 = trunc i32 %add5.lcssa to i8
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i8 [ %conv6, %for.cond.for.cond.cleanup_crit_edge ], [ 0, %entry ]
  ret i8 %sum.0.lcssa

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %sum.013 = phi i32 [ %add5, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %conv4 = and i32 %sum.013, 255
  %add = add nuw nsw i32 %conv, %conv4
  %add5 = add nuw nsw i32 %add, %conv3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}

; CHECK-LABEL: @reduction_i16_1
;
; short reduction_i16_1(short *a, short *b, int n) {
;   short sum = 0;
;   for (int i = 0; i < n; ++i)
;     sum += (a[i] + b[i]);
;   return sum;
; }
;
; CHECK: vector.body:
; CHECK:   phi <8 x i16>
; CHECK:   load <8 x i16>
; CHECK:   load <8 x i16>
; CHECK:   add <8 x i16>
; CHECK:   add <8 x i16>
;
; CHECK: middle.block:
; CHECK:   [[Rdx:%[a-zA-Z0-9.]+]] = call i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16>
; CHECK:   zext i16 [[Rdx]] to i32
;
define i16 @reduction_i16_1(i16* nocapture readonly %a, i16* nocapture readonly %b, i32 %n) {
entry:
  %cmp.16 = icmp sgt i32 %n, 0
  br i1 %cmp.16, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  %add5.lcssa = phi i32 [ %add5, %for.body ]
  %conv6 = trunc i32 %add5.lcssa to i16
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i16 [ %conv6, %for.cond.for.cond.cleanup_crit_edge ], [ 0, %entry ]
  ret i16 %sum.0.lcssa

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %sum.017 = phi i32 [ %add5, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %a, i64 %indvars.iv
  %0 = load i16, i16* %arrayidx, align 2
  %conv.14 = zext i16 %0 to i32
  %arrayidx2 = getelementptr inbounds i16, i16* %b, i64 %indvars.iv
  %1 = load i16, i16* %arrayidx2, align 2
  %conv3.15 = zext i16 %1 to i32
  %conv4.13 = and i32 %sum.017, 65535
  %add = add nuw nsw i32 %conv.14, %conv4.13
  %add5 = add nuw nsw i32 %add, %conv3.15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}

; CHECK-LABEL: @reduction_i16_2
;
; short reduction_i16_2(char *a, char *b, int n) {
;   short sum = 0;
;   for (int i = 0; i < n; ++i)
;     sum += (a[i] + b[i]);
;   return sum;
; }
;
; CHECK: vector.body:
; CHECK:   phi <8 x i16>
; CHECK:   [[Ld1:%[a-zA-Z0-9.]+]] = load <8 x i8>
; CHECK:   zext <8 x i8> [[Ld1]] to <8 x i16>
; CHECK:   [[Ld2:%[a-zA-Z0-9.]+]] = load <8 x i8>
; CHECK:   zext <8 x i8> [[Ld2]] to <8 x i16>
; CHECK:   add <8 x i16>
; CHECK:   add <8 x i16>
;
; CHECK: middle.block:
; CHECK:   [[Rdx:%[a-zA-Z0-9.]+]] = call i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16>
; CHECK:   zext i16 [[Rdx]] to i32
;
define i16 @reduction_i16_2(i8* nocapture readonly %a, i8* nocapture readonly %b, i32 %n) {
entry:
  %cmp.14 = icmp sgt i32 %n, 0
  br i1 %cmp.14, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  %add5.lcssa = phi i32 [ %add5, %for.body ]
  %conv6 = trunc i32 %add5.lcssa to i16
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i16 [ %conv6, %for.cond.for.cond.cleanup_crit_edge ], [ 0, %entry ]
  ret i16 %sum.0.lcssa

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %sum.015 = phi i32 [ %add5, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds i8, i8* %b, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %conv4.13 = and i32 %sum.015, 65535
  %add = add nuw nsw i32 %conv, %conv4.13
  %add5 = add nuw nsw i32 %add, %conv3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}
