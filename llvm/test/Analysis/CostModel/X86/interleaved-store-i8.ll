; REQUIRES: asserts
; RUN: opt -loop-vectorize -S -mcpu=core-avx2 --debug-only=loop-vectorize -vectorizer-maximize-bandwidth < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @doit_stride3(i8* nocapture %Ptr, i32 %Nels) local_unnamed_addr {
;CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %conv4
;CHECK: LV: Found an estimated cost of 8 for VF 2 For instruction:   store i8 %conv4
;CHECK: LV: Found an estimated cost of 9 for VF 4 For instruction:   store i8 %conv4
;CHECK: LV: Found an estimated cost of 12 for VF 8 For instruction:   store i8 %conv4
;CHECK: LV: Found an estimated cost of 13 for VF 16 For instruction:   store i8 %conv4
;CHECK: LV: Found an estimated cost of 16 for VF 32 For instruction:   store i8 %conv4
entry:
  %cmp14 = icmp sgt i32 %Nels, 0
  br i1 %cmp14, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %conv = trunc i32 %Nels to i8
  %conv1 = shl i8 %conv, 1
  %conv4 = shl i8 %conv, 2
  br label %for.body

for.body:
  %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %Ptr.addr.015 = phi i8* [ %Ptr, %for.body.lr.ph ], [ %incdec.ptr5, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, i8* %Ptr.addr.015, i64 1
  store i8 %conv, i8* %Ptr.addr.015, align 1
  %incdec.ptr2 = getelementptr inbounds i8, i8* %Ptr.addr.015, i64 2
  store i8 %conv1, i8* %incdec.ptr, align 1
  %incdec.ptr5 = getelementptr inbounds i8, i8* %Ptr.addr.015, i64 3
  store i8 %conv4, i8* %incdec.ptr2, align 1
  %inc = add nuw nsw i32 %i.016, 1
  %exitcond = icmp eq i32 %inc, %Nels
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @doit_stride4(i8* nocapture %Ptr, i32 %Nels) local_unnamed_addr {
;CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %conv7
;CHECK: LV: Found an estimated cost of 13 for VF 2 For instruction:   store i8 %conv7
;CHECK: LV: Found an estimated cost of 10 for VF 4 For instruction:   store i8 %conv7
;CHECK: LV: Found an estimated cost of 11 for VF 8 For instruction:   store i8 %conv7
;CHECK: LV: Found an estimated cost of 12 for VF 16 For instruction:   store i8 %conv7
;CHECK: LV: Found an estimated cost of 16 for VF 32 For instruction:   store i8 %conv7
entry:
  %cmp19 = icmp sgt i32 %Nels, 0
  br i1 %cmp19, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %conv = trunc i32 %Nels to i8
  %conv1 = shl i8 %conv, 1
  %conv4 = shl i8 %conv, 2
  %mul6 = mul nsw i32 %Nels, 5
  %conv7 = trunc i32 %mul6 to i8
  br label %for.body

for.body:
  %i.021 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %Ptr.addr.020 = phi i8* [ %Ptr, %for.body.lr.ph ], [ %incdec.ptr8, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, i8* %Ptr.addr.020, i64 1
  store i8 %conv, i8* %Ptr.addr.020, align 1
  %incdec.ptr2 = getelementptr inbounds i8, i8* %Ptr.addr.020, i64 2
  store i8 %conv1, i8* %incdec.ptr, align 1
  %incdec.ptr5 = getelementptr inbounds i8, i8* %Ptr.addr.020, i64 3
  store i8 %conv4, i8* %incdec.ptr2, align 1
  %incdec.ptr8 = getelementptr inbounds i8, i8* %Ptr.addr.020, i64 4
  store i8 %conv7, i8* %incdec.ptr5, align 1
  %inc = add nuw nsw i32 %i.021, 1
  %exitcond = icmp eq i32 %inc, %Nels
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
