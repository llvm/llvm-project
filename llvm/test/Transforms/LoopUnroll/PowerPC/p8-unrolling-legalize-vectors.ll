; RUN: opt < %s -S -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -loop-unroll | FileCheck %s
; RUN: opt < %s -S -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -loop-unroll | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: norecurse nounwind
define i8* @f(i8* returned %s, i32 zeroext %x, i32 signext %k) local_unnamed_addr #0 {
entry:
  %cmp10 = icmp sgt i32 %k, 0
  br i1 %cmp10, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %wide.trip.count = zext i32 %k to i64
  %min.iters.check = icmp ult i32 %k, 16
  br i1 %min.iters.check, label %for.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i64 %wide.trip.count, 4294967280
  %broadcast.splatinsert = insertelement <16 x i32> undef, i32 %x, i32 0
  %broadcast.splat = shufflevector <16 x i32> %broadcast.splatinsert, <16 x i32> undef, <16 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind12 = phi <16 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, %vector.ph ], [ %vec.ind.next13, %vector.body ]
  %0 = shl <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, %vec.ind12
  %1 = and <16 x i32> %0, %broadcast.splat
  %2 = icmp eq <16 x i32> %1, zeroinitializer
  %3 = select <16 x i1> %2, <16 x i8> <i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48>, <16 x i8> <i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49, i8 49>
  %4 = getelementptr inbounds i8, i8* %s, i64 %index
  %5 = bitcast i8* %4 to <16 x i8>*
  store <16 x i8> %3, <16 x i8>* %5, align 1
  %index.next = add i64 %index, 16
  %vec.ind.next13 = add <16 x i32> %vec.ind12, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %6 = icmp eq i64 %index.next, %n.vec
  br i1 %6, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %middle.block, %for.body.lr.ph
  %indvars.iv.ph = phi i64 [ 0, %for.body.lr.ph ], [ %n.vec, %middle.block ]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader ]
  %7 = trunc i64 %indvars.iv to i32
  %shl = shl i32 1, %7
  %and = and i32 %shl, %x
  %tobool = icmp eq i32 %and, 0
  %conv = select i1 %tobool, i8 48, i8 49
  %arrayidx = getelementptr inbounds i8, i8* %s, i64 %indvars.iv
  store i8 %conv, i8* %arrayidx, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %middle.block, %entry
  %idxprom1 = sext i32 %k to i64
  %arrayidx2 = getelementptr inbounds i8, i8* %s, i64 %idxprom1
  store i8 0, i8* %arrayidx2, align 1
  ret i8* %s
}


; CHECK-LABEL: vector.body
; CHECK:      shl
; CHECK-NEXT: and
; CHECK: shl
; CHECK-NEXT: and
; CHECK: label %vector.body

