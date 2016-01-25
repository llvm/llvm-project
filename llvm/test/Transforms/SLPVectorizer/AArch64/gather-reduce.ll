; RUN: opt -S -slp-vectorizer -dce -instcombine < %s | FileCheck %s --check-prefix=PROFITABLE
; RUN: opt -S -slp-vectorizer -slp-threshold=-12 -dce -instcombine < %s | FileCheck %s --check-prefix=UNPROFITABLE

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; These tests check that we vectorize the index calculations in the
; gather-reduce pattern shown below. We check cases having i32 and i64
; subtraction.
;
; int gather_reduce_8x16(short *a, short *b, short *g, int n) {
;   int sum = 0;
;   for (int i = 0; i < n ; ++i) {
;     sum += g[*a++ - b[0]]; sum += g[*a++ - b[1]];
;     sum += g[*a++ - b[2]]; sum += g[*a++ - b[3]];
;     sum += g[*a++ - b[4]]; sum += g[*a++ - b[5]];
;     sum += g[*a++ - b[6]]; sum += g[*a++ - b[7]];
;   }
;   return sum;
; }

; PROFITABLE-LABEL: @gather_reduce_8x16_i32
;
; PROFITABLE: [[L:%[a-zA-Z0-9.]+]] = load <8 x i16>
; PROFITABLE: zext <8 x i16> [[L]] to <8 x i32>
; PROFITABLE: [[S:%[a-zA-Z0-9.]+]] = sub nsw <8 x i32>
; PROFITABLE: [[X:%[a-zA-Z0-9.]+]] = extractelement <8 x i32> [[S]]
; PROFITABLE: sext i32 [[X]] to i64
;
define i32 @gather_reduce_8x16_i32(i16* nocapture readonly %a, i16* nocapture readonly %b, i16* nocapture readonly %g, i32 %n) {
entry:
  %cmp.99 = icmp sgt i32 %n, 0
  br i1 %cmp.99, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add66, %for.cond.cleanup.loopexit ]
  ret i32 %sum.0.lcssa

for.body:
  %i.0103 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %sum.0102 = phi i32 [ %add66, %for.body ], [ 0, %for.body.preheader ]
  %a.addr.0101 = phi i16* [ %incdec.ptr58, %for.body ], [ %a, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds i16, i16* %a.addr.0101, i64 1
  %0 = load i16, i16* %a.addr.0101, align 2
  %conv = zext i16 %0 to i32
  %incdec.ptr1 = getelementptr inbounds i16, i16* %b, i64 1
  %1 = load i16, i16* %b, align 2
  %conv2 = zext i16 %1 to i32
  %sub = sub nsw i32 %conv, %conv2
  %arrayidx = getelementptr inbounds i16, i16* %g, i32 %sub
  %2 = load i16, i16* %arrayidx, align 2
  %conv3 = zext i16 %2 to i32
  %add = add nsw i32 %conv3, %sum.0102
  %incdec.ptr4 = getelementptr inbounds i16, i16* %a.addr.0101, i64 2
  %3 = load i16, i16* %incdec.ptr, align 2
  %conv5 = zext i16 %3 to i32
  %incdec.ptr6 = getelementptr inbounds i16, i16* %b, i64 2
  %4 = load i16, i16* %incdec.ptr1, align 2
  %conv7 = zext i16 %4 to i32
  %sub8 = sub nsw i32 %conv5, %conv7
  %arrayidx10 = getelementptr inbounds i16, i16* %g, i32 %sub8
  %5 = load i16, i16* %arrayidx10, align 2
  %conv11 = zext i16 %5 to i32
  %add12 = add nsw i32 %add, %conv11
  %incdec.ptr13 = getelementptr inbounds i16, i16* %a.addr.0101, i64 3
  %6 = load i16, i16* %incdec.ptr4, align 2
  %conv14 = zext i16 %6 to i32
  %incdec.ptr15 = getelementptr inbounds i16, i16* %b, i64 3
  %7 = load i16, i16* %incdec.ptr6, align 2
  %conv16 = zext i16 %7 to i32
  %sub17 = sub nsw i32 %conv14, %conv16
  %arrayidx19 = getelementptr inbounds i16, i16* %g, i32 %sub17
  %8 = load i16, i16* %arrayidx19, align 2
  %conv20 = zext i16 %8 to i32
  %add21 = add nsw i32 %add12, %conv20
  %incdec.ptr22 = getelementptr inbounds i16, i16* %a.addr.0101, i64 4
  %9 = load i16, i16* %incdec.ptr13, align 2
  %conv23 = zext i16 %9 to i32
  %incdec.ptr24 = getelementptr inbounds i16, i16* %b, i64 4
  %10 = load i16, i16* %incdec.ptr15, align 2
  %conv25 = zext i16 %10 to i32
  %sub26 = sub nsw i32 %conv23, %conv25
  %arrayidx28 = getelementptr inbounds i16, i16* %g, i32 %sub26
  %11 = load i16, i16* %arrayidx28, align 2
  %conv29 = zext i16 %11 to i32
  %add30 = add nsw i32 %add21, %conv29
  %incdec.ptr31 = getelementptr inbounds i16, i16* %a.addr.0101, i64 5
  %12 = load i16, i16* %incdec.ptr22, align 2
  %conv32 = zext i16 %12 to i32
  %incdec.ptr33 = getelementptr inbounds i16, i16* %b, i64 5
  %13 = load i16, i16* %incdec.ptr24, align 2
  %conv34 = zext i16 %13 to i32
  %sub35 = sub nsw i32 %conv32, %conv34
  %arrayidx37 = getelementptr inbounds i16, i16* %g, i32 %sub35
  %14 = load i16, i16* %arrayidx37, align 2
  %conv38 = zext i16 %14 to i32
  %add39 = add nsw i32 %add30, %conv38
  %incdec.ptr40 = getelementptr inbounds i16, i16* %a.addr.0101, i64 6
  %15 = load i16, i16* %incdec.ptr31, align 2
  %conv41 = zext i16 %15 to i32
  %incdec.ptr42 = getelementptr inbounds i16, i16* %b, i64 6
  %16 = load i16, i16* %incdec.ptr33, align 2
  %conv43 = zext i16 %16 to i32
  %sub44 = sub nsw i32 %conv41, %conv43
  %arrayidx46 = getelementptr inbounds i16, i16* %g, i32 %sub44
  %17 = load i16, i16* %arrayidx46, align 2
  %conv47 = zext i16 %17 to i32
  %add48 = add nsw i32 %add39, %conv47
  %incdec.ptr49 = getelementptr inbounds i16, i16* %a.addr.0101, i64 7
  %18 = load i16, i16* %incdec.ptr40, align 2
  %conv50 = zext i16 %18 to i32
  %incdec.ptr51 = getelementptr inbounds i16, i16* %b, i64 7
  %19 = load i16, i16* %incdec.ptr42, align 2
  %conv52 = zext i16 %19 to i32
  %sub53 = sub nsw i32 %conv50, %conv52
  %arrayidx55 = getelementptr inbounds i16, i16* %g, i32 %sub53
  %20 = load i16, i16* %arrayidx55, align 2
  %conv56 = zext i16 %20 to i32
  %add57 = add nsw i32 %add48, %conv56
  %incdec.ptr58 = getelementptr inbounds i16, i16* %a.addr.0101, i64 8
  %21 = load i16, i16* %incdec.ptr49, align 2
  %conv59 = zext i16 %21 to i32
  %22 = load i16, i16* %incdec.ptr51, align 2
  %conv61 = zext i16 %22 to i32
  %sub62 = sub nsw i32 %conv59, %conv61
  %arrayidx64 = getelementptr inbounds i16, i16* %g, i32 %sub62
  %23 = load i16, i16* %arrayidx64, align 2
  %conv65 = zext i16 %23 to i32
  %add66 = add nsw i32 %add57, %conv65
  %inc = add nuw nsw i32 %i.0103, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

; UNPROFITABLE-LABEL: @gather_reduce_8x16_i64
;
; UNPROFITABLE: [[L:%[a-zA-Z0-9.]+]] = load <8 x i16>
; UNPROFITABLE: zext <8 x i16> [[L]] to <8 x i32>
; UNPROFITABLE: [[S:%[a-zA-Z0-9.]+]] = sub nsw <8 x i32>
; UNPROFITABLE: [[X:%[a-zA-Z0-9.]+]] = extractelement <8 x i32> [[S]]
; UNPROFITABLE: sext i32 [[X]] to i64
;
; TODO: Although we can now vectorize this case while converting the i64
;       subtractions to i32, the cost model currently finds vectorization to be
;       unprofitable. The cost model is penalizing the sign and zero
;       extensions in the vectorized version, but they are actually free.
;
define i32 @gather_reduce_8x16_i64(i16* nocapture readonly %a, i16* nocapture readonly %b, i16* nocapture readonly %g, i32 %n) {
entry:
  %cmp.99 = icmp sgt i32 %n, 0
  br i1 %cmp.99, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add66, %for.cond.cleanup.loopexit ]
  ret i32 %sum.0.lcssa

for.body:
  %i.0103 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %sum.0102 = phi i32 [ %add66, %for.body ], [ 0, %for.body.preheader ]
  %a.addr.0101 = phi i16* [ %incdec.ptr58, %for.body ], [ %a, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds i16, i16* %a.addr.0101, i64 1
  %0 = load i16, i16* %a.addr.0101, align 2
  %conv = zext i16 %0 to i64
  %incdec.ptr1 = getelementptr inbounds i16, i16* %b, i64 1
  %1 = load i16, i16* %b, align 2
  %conv2 = zext i16 %1 to i64
  %sub = sub nsw i64 %conv, %conv2
  %arrayidx = getelementptr inbounds i16, i16* %g, i64 %sub
  %2 = load i16, i16* %arrayidx, align 2
  %conv3 = zext i16 %2 to i32
  %add = add nsw i32 %conv3, %sum.0102
  %incdec.ptr4 = getelementptr inbounds i16, i16* %a.addr.0101, i64 2
  %3 = load i16, i16* %incdec.ptr, align 2
  %conv5 = zext i16 %3 to i64
  %incdec.ptr6 = getelementptr inbounds i16, i16* %b, i64 2
  %4 = load i16, i16* %incdec.ptr1, align 2
  %conv7 = zext i16 %4 to i64
  %sub8 = sub nsw i64 %conv5, %conv7
  %arrayidx10 = getelementptr inbounds i16, i16* %g, i64 %sub8
  %5 = load i16, i16* %arrayidx10, align 2
  %conv11 = zext i16 %5 to i32
  %add12 = add nsw i32 %add, %conv11
  %incdec.ptr13 = getelementptr inbounds i16, i16* %a.addr.0101, i64 3
  %6 = load i16, i16* %incdec.ptr4, align 2
  %conv14 = zext i16 %6 to i64
  %incdec.ptr15 = getelementptr inbounds i16, i16* %b, i64 3
  %7 = load i16, i16* %incdec.ptr6, align 2
  %conv16 = zext i16 %7 to i64
  %sub17 = sub nsw i64 %conv14, %conv16
  %arrayidx19 = getelementptr inbounds i16, i16* %g, i64 %sub17
  %8 = load i16, i16* %arrayidx19, align 2
  %conv20 = zext i16 %8 to i32
  %add21 = add nsw i32 %add12, %conv20
  %incdec.ptr22 = getelementptr inbounds i16, i16* %a.addr.0101, i64 4
  %9 = load i16, i16* %incdec.ptr13, align 2
  %conv23 = zext i16 %9 to i64
  %incdec.ptr24 = getelementptr inbounds i16, i16* %b, i64 4
  %10 = load i16, i16* %incdec.ptr15, align 2
  %conv25 = zext i16 %10 to i64
  %sub26 = sub nsw i64 %conv23, %conv25
  %arrayidx28 = getelementptr inbounds i16, i16* %g, i64 %sub26
  %11 = load i16, i16* %arrayidx28, align 2
  %conv29 = zext i16 %11 to i32
  %add30 = add nsw i32 %add21, %conv29
  %incdec.ptr31 = getelementptr inbounds i16, i16* %a.addr.0101, i64 5
  %12 = load i16, i16* %incdec.ptr22, align 2
  %conv32 = zext i16 %12 to i64
  %incdec.ptr33 = getelementptr inbounds i16, i16* %b, i64 5
  %13 = load i16, i16* %incdec.ptr24, align 2
  %conv34 = zext i16 %13 to i64
  %sub35 = sub nsw i64 %conv32, %conv34
  %arrayidx37 = getelementptr inbounds i16, i16* %g, i64 %sub35
  %14 = load i16, i16* %arrayidx37, align 2
  %conv38 = zext i16 %14 to i32
  %add39 = add nsw i32 %add30, %conv38
  %incdec.ptr40 = getelementptr inbounds i16, i16* %a.addr.0101, i64 6
  %15 = load i16, i16* %incdec.ptr31, align 2
  %conv41 = zext i16 %15 to i64
  %incdec.ptr42 = getelementptr inbounds i16, i16* %b, i64 6
  %16 = load i16, i16* %incdec.ptr33, align 2
  %conv43 = zext i16 %16 to i64
  %sub44 = sub nsw i64 %conv41, %conv43
  %arrayidx46 = getelementptr inbounds i16, i16* %g, i64 %sub44
  %17 = load i16, i16* %arrayidx46, align 2
  %conv47 = zext i16 %17 to i32
  %add48 = add nsw i32 %add39, %conv47
  %incdec.ptr49 = getelementptr inbounds i16, i16* %a.addr.0101, i64 7
  %18 = load i16, i16* %incdec.ptr40, align 2
  %conv50 = zext i16 %18 to i64
  %incdec.ptr51 = getelementptr inbounds i16, i16* %b, i64 7
  %19 = load i16, i16* %incdec.ptr42, align 2
  %conv52 = zext i16 %19 to i64
  %sub53 = sub nsw i64 %conv50, %conv52
  %arrayidx55 = getelementptr inbounds i16, i16* %g, i64 %sub53
  %20 = load i16, i16* %arrayidx55, align 2
  %conv56 = zext i16 %20 to i32
  %add57 = add nsw i32 %add48, %conv56
  %incdec.ptr58 = getelementptr inbounds i16, i16* %a.addr.0101, i64 8
  %21 = load i16, i16* %incdec.ptr49, align 2
  %conv59 = zext i16 %21 to i64
  %22 = load i16, i16* %incdec.ptr51, align 2
  %conv61 = zext i16 %22 to i64
  %sub62 = sub nsw i64 %conv59, %conv61
  %arrayidx64 = getelementptr inbounds i16, i16* %g, i64 %sub62
  %23 = load i16, i16* %arrayidx64, align 2
  %conv65 = zext i16 %23 to i32
  %add66 = add nsw i32 %add57, %conv65
  %inc = add nuw nsw i32 %i.0103, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
