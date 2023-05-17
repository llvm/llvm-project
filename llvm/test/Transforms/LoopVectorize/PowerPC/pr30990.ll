; RUN: opt < %s -passes=loop-vectorize -mcpu=pwr8 -mattr=+vsx -force-vector-interleave=1 -vectorizer-maximize-bandwidth=0 -S | FileCheck %s

target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @foo(ptr readonly %ptr, i32 signext %l) {
entry:
  %idx.ext = sext i32 %l to i64
  %add.ptr = getelementptr inbounds i8, ptr %ptr, i64 %idx.ext
  %cmp7 = icmp sgt i32 %l, 0
  br i1 %cmp7, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %count.09 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %ptr.addr.08 = phi ptr [ %incdec.ptr, %while.body ], [ %ptr, %while.body.preheader ]
  %0 = load i8, ptr %ptr.addr.08, align 1
  %cmp1 = icmp slt i8 %0, -64
  %cond = zext i1 %cmp1 to i32
  %add = add nsw i32 %cond, %count.09
  %incdec.ptr = getelementptr inbounds i8, ptr %ptr.addr.08, i64 1
  %cmp = icmp ult ptr %incdec.ptr, %add.ptr
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi i32 [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %count.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %while.end.loopexit ]
  ret i32 %count.0.lcssa

; CHECK: load <4 x i8>
; CHECK: icmp slt <4 x i8>
}


define signext i16 @foo2(ptr readonly %ptr, i32 signext %l) {
entry:
  %idx.ext = sext i32 %l to i64
  %add.ptr = getelementptr inbounds i8, ptr %ptr, i64 %idx.ext
  %cmp7 = icmp sgt i32 %l, 0
  br i1 %cmp7, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %count.09 = phi i16 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %ptr.addr.08 = phi ptr [ %incdec.ptr, %while.body ], [ %ptr, %while.body.preheader ]
  %0 = load i8, ptr %ptr.addr.08, align 1
  %cmp1 = icmp slt i8 %0, -64
  %cond = zext i1 %cmp1 to i16
  %add = add nsw i16 %cond, %count.09
  %incdec.ptr = getelementptr inbounds i8, ptr %ptr.addr.08, i64 1
  %cmp = icmp ult ptr %incdec.ptr, %add.ptr
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi i16 [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %count.0.lcssa = phi i16 [ 0, %entry ], [ %add.lcssa, %while.end.loopexit ]
  ret i16 %count.0.lcssa

; CHECK-LABEL: foo2
; CHECK: load <8 x i8>
; CHECK: icmp slt <8 x i8>
}

define signext i32 @foo3(ptr readonly %ptr, i32 signext %l) {
entry:
  %idx.ext = sext i32 %l to i64
  %add.ptr = getelementptr inbounds i16, ptr %ptr, i64 %idx.ext
  %cmp7 = icmp sgt i32 %l, 0
  br i1 %cmp7, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %count.09 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %ptr.addr.16 = phi ptr [ %incdec.ptr, %while.body ], [ %ptr, %while.body.preheader ]
  %0 = load i16, ptr %ptr.addr.16, align 1
  %cmp1 = icmp slt i16 %0, -64
  %cond = zext i1 %cmp1 to i32
  %add = add nsw i32 %cond, %count.09
  %incdec.ptr = getelementptr inbounds i16, ptr %ptr.addr.16, i64 1
  %cmp = icmp ult ptr %incdec.ptr, %add.ptr
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi i32 [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %count.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %while.end.loopexit ]
  ret i32 %count.0.lcssa

; CHECK-LABEL: foo3
; CHECK: load <4 x i16>
; CHECK: icmp slt <4 x i16>
}

define i64 @foo4(ptr readonly %ptr, i32 signext %l) {
entry:
  %idx.ext = sext i32 %l to i64
  %add.ptr = getelementptr inbounds i16, ptr %ptr, i64 %idx.ext
  %cmp7 = icmp sgt i32 %l, 0
  br i1 %cmp7, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %count.09 = phi i64 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %ptr.addr.16 = phi ptr [ %incdec.ptr, %while.body ], [ %ptr, %while.body.preheader ]
  %0 = load i16, ptr %ptr.addr.16, align 1
  %cmp1 = icmp slt i16 %0, -64
  %cond = zext i1 %cmp1 to i64
  %add = add nsw i64 %cond, %count.09
  %incdec.ptr = getelementptr inbounds i16, ptr %ptr.addr.16, i64 1
  %cmp = icmp ult ptr %incdec.ptr, %add.ptr
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi i64 [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %count.0.lcssa = phi i64 [ 0, %entry ], [ %add.lcssa, %while.end.loopexit ]
  ret i64 %count.0.lcssa

; CHECK-LABEL: foo4
; CHECK: load <2 x i16>
; CHECK: icmp slt <2 x i16>
}

