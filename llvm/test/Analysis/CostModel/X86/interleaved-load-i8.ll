; REQUIRES: asserts
; RUN: opt -loop-vectorize -S -mcpu=core-avx2 --debug-only=loop-vectorize -vectorizer-maximize-bandwidth < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @doit_stride3(i8* nocapture readonly %Ptr, i32 %Nels)  {
;CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 11 for VF 2 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 5 for VF 4 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 10 for VF 8 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 13 for VF 16 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 16 for VF 32 For instruction:   %0 = load i8
entry:
  %cmp13 = icmp sgt i32 %Nels, 0
  br i1 %cmp13, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %Ptr.addr.016 = phi i8* [ %incdec.ptr2, %for.body ], [ %Ptr, %for.body.preheader ]
  %i.015 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %s.014 = phi i32 [ %add6, %for.body ], [ 0, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, i8* %Ptr.addr.016, i64 1
  %0 = load i8, i8* %Ptr.addr.016, align 1
  %incdec.ptr1 = getelementptr inbounds i8, i8* %Ptr.addr.016, i64 2
  %1 = load i8, i8* %incdec.ptr, align 1
  %incdec.ptr2 = getelementptr inbounds i8, i8* %Ptr.addr.016, i64 3
  %2 = load i8, i8* %incdec.ptr1, align 1
  %conv = zext i8 %0 to i32
  %conv3 = zext i8 %1 to i32
  %conv4 = zext i8 %2 to i32
  %add = add i32 %s.014, %conv
  %add5 = add i32 %add, %conv3
  %add6 = add i32 %add5, %conv4
  %inc = add nuw nsw i32 %i.015, 1
  %exitcond = icmp eq i32 %inc, %Nels
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  %add6.lcssa = phi i32 [ %add6, %for.body ]
  br label %for.end

for.end:
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %add6.lcssa, %for.end.loopexit ]
  ret i32 %s.0.lcssa
}

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @doit_stride4(i8* nocapture readonly %Ptr, i32 %Nels) local_unnamed_addr {
;CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 13 for VF 2 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 5 for VF 4 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 21 for VF 8 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 41 for VF 16 For instruction:   %0 = load i8
;CHECK: LV: Found an estimated cost of 84 for VF 32 For instruction:   %0 = load i8
entry:
  %cmp59 = icmp sgt i32 %Nels, 0
  br i1 %cmp59, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %Ptr.addr.062 = phi i8* [ %incdec.ptr3, %for.body ], [ %Ptr, %for.body.preheader ]
  %i.061 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %s.060 = phi i32 [ %cond39, %for.body ], [ 0, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, i8* %Ptr.addr.062, i64 1
  %0 = load i8, i8* %Ptr.addr.062, align 1
  %incdec.ptr1 = getelementptr inbounds i8, i8* %Ptr.addr.062, i64 2
  %1 = load i8, i8* %incdec.ptr, align 1
  %incdec.ptr2 = getelementptr inbounds i8, i8* %Ptr.addr.062, i64 3
  %2 = load i8, i8* %incdec.ptr1, align 1
  %incdec.ptr3 = getelementptr inbounds i8, i8* %Ptr.addr.062, i64 4
  %3 = load i8, i8* %incdec.ptr2, align 1
  %cmp5 = icmp ult i8 %0, %1
  %.sink = select i1 %cmp5, i8 %0, i8 %1
  %cmp12 = icmp ult i8 %.sink, %2
  %.sink40 = select i1 %cmp12, i8 %.sink, i8 %2
  %cmp23 = icmp ult i8 %.sink40, %3
  %.sink41 = select i1 %cmp23, i8 %.sink40, i8 %3
  %conv28 = zext i8 %.sink41 to i32
  %cmp33 = icmp slt i32 %s.060, %conv28
  %cond39 = select i1 %cmp33, i32 %s.060, i32 %conv28
  %inc = add nuw nsw i32 %i.061, 1
  %exitcond = icmp eq i32 %inc, %Nels
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit: 
  %cond39.lcssa = phi i32 [ %cond39, %for.body ]
  br label %for.end

for.end:
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %cond39.lcssa, %for.end.loopexit ]
  ret i32 %s.0.lcssa
}
