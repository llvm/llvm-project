; RUN: opt -passes=loop-vectorize < %s -S -o - | FileCheck %s --check-prefixes=CHECK,CHECK-2,CHECK-NO4
; RUN: opt -passes=loop-vectorize -mve-max-interleave-factor=1 < %s -S -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NO2,CHECK-NO4
; RUN: opt -passes=loop-vectorize -mve-max-interleave-factor=2 < %s -S -o - | FileCheck %s --check-prefixes=CHECK,CHECK-2,CHECK-NO4
; RUN: opt -passes=loop-vectorize -mve-max-interleave-factor=4 < %s -S -o - | FileCheck %s --check-prefixes=CHECK,CHECK-2,CHECK-4

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-none-none-eabi"

; CHECK-LABEL: vld2
; CHECK-2: vector.body
; CHECK-NO2-NOT: vector.body
define void @vld2(half* nocapture readonly %pIn, half* nocapture %pOut, i32 %numRows, i32 %numCols, i32 %scale.coerce) #0 {
entry:
  %tmp.0.extract.trunc = trunc i32 %scale.coerce to i16
  %0 = bitcast i16 %tmp.0.extract.trunc to half
  %mul = mul i32 %numCols, %numRows
  %shr = lshr i32 %mul, 2
  %cmp26 = icmp eq i32 %shr, 0
  br i1 %cmp26, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %pIn.addr.029 = phi half* [ %add.ptr, %while.body ], [ %pIn, %entry ]
  %pOut.addr.028 = phi half* [ %add.ptr7, %while.body ], [ %pOut, %entry ]
  %blkCnt.027 = phi i32 [ %dec, %while.body ], [ %shr, %entry ]
  %1 = load half, half* %pIn.addr.029, align 2
  %arrayidx2 = getelementptr inbounds half, half* %pIn.addr.029, i32 1
  %2 = load half, half* %arrayidx2, align 2
  %mul3 = fmul half %1, %0
  %mul4 = fmul half %2, %0
  store half %mul3, half* %pOut.addr.028, align 2
  %arrayidx6 = getelementptr inbounds half, half* %pOut.addr.028, i32 1
  store half %mul4, half* %arrayidx6, align 2
  %add.ptr = getelementptr inbounds half, half* %pIn.addr.029, i32 2
  %add.ptr7 = getelementptr inbounds half, half* %pOut.addr.028, i32 2
  %dec = add nsw i32 %blkCnt.027, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

; CHECK-LABEL: vld4
; CHECK-4: vector.body
; CHECK-NO4-NOT: vector.body
define void @vld4(half* nocapture readonly %pIn, half* nocapture %pOut, i32 %numRows, i32 %numCols, i32 %scale.coerce) #0 {
entry:
  %tmp.0.extract.trunc = trunc i32 %scale.coerce to i16
  %0 = bitcast i16 %tmp.0.extract.trunc to half
  %mul = mul i32 %numCols, %numRows
  %shr = lshr i32 %mul, 2
  %cmp38 = icmp eq i32 %shr, 0
  br i1 %cmp38, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %pIn.addr.041 = phi half* [ %add.ptr, %while.body ], [ %pIn, %entry ]
  %pOut.addr.040 = phi half* [ %add.ptr13, %while.body ], [ %pOut, %entry ]
  %blkCnt.039 = phi i32 [ %dec, %while.body ], [ %shr, %entry ]
  %1 = load half, half* %pIn.addr.041, align 2
  %arrayidx2 = getelementptr inbounds half, half* %pIn.addr.041, i32 1
  %2 = load half, half* %arrayidx2, align 2
  %arrayidx3 = getelementptr inbounds half, half* %pIn.addr.041, i32 2
  %3 = load half, half* %arrayidx3, align 2
  %arrayidx4 = getelementptr inbounds half, half* %pIn.addr.041, i32 3
  %4 = load half, half* %arrayidx4, align 2
  %mul5 = fmul half %1, %0
  %mul6 = fmul half %2, %0
  %mul7 = fmul half %3, %0
  %mul8 = fmul half %4, %0
  store half %mul5, half* %pOut.addr.040, align 2
  %arrayidx10 = getelementptr inbounds half, half* %pOut.addr.040, i32 1
  store half %mul6, half* %arrayidx10, align 2
  %arrayidx11 = getelementptr inbounds half, half* %pOut.addr.040, i32 2
  store half %mul7, half* %arrayidx11, align 2
  %arrayidx12 = getelementptr inbounds half, half* %pOut.addr.040, i32 3
  store half %mul8, half* %arrayidx12, align 2
  %add.ptr = getelementptr inbounds half, half* %pIn.addr.041, i32 4
  %add.ptr13 = getelementptr inbounds half, half* %pOut.addr.040, i32 4
  %dec = add nsw i32 %blkCnt.039, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

attributes #0 = { "target-features"="+armv8.1-m.main,+fp-armv8d16,+fp-armv8d16sp,+fp16,+fp64,+fullfp16,+hwdiv,+lob,+mve.fp,+ras,+strict-align,+thumb-mode,+vfp2,+vfp2sp,+vfp3d16,+vfp3d16sp,+vfp4d16,+vfp4d16sp,-crypto,-d32,-fp-armv8,-fp-armv8sp,-neon,-vfp3,-vfp3sp,-vfp4,-vfp4sp" }
