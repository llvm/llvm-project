; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S -enable-if-conversion | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK: fc
;CHECK: load <4 x i16>
;CHECK-NEXT: shufflevector <4 x i16>
;CHECK: select <4 x i1>
;CHECK: store <4 x i16>
;CHECK: ret
define void @fc(i16* nocapture %p, i32 %n, i32 %size) nounwind uwtable ssp {
entry:
  br label %do.body

do.body:                                          ; preds = %cond.end, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %cond.end ]
  %p.addr.0 = phi i16* [ %p, %entry ], [ %incdec.ptr, %cond.end ]
  %incdec.ptr = getelementptr inbounds i16, i16* %p.addr.0, i64 -1
  %0 = load i16, i16* %incdec.ptr, align 2
  %conv = zext i16 %0 to i32
  %cmp = icmp ult i32 %conv, %size
  br i1 %cmp, label %cond.end, label %cond.true

cond.true:                                        ; preds = %do.body
  %sub = sub i32 %conv, %size
  %phitmp = trunc i32 %sub to i16
  br label %cond.end

cond.end:                                         ; preds = %do.body, %cond.true
  %cond = phi i16 [ %phitmp, %cond.true ], [ 0, %do.body ]
  store i16 %cond, i16* %incdec.ptr, align 2
  %dec = add i32 %n.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:                                           ; preds = %cond.end
  ret void
}

;CHECK: example1
;CHECK: load <4 x i32>
;CHECK-NEXT: shufflevector <4 x i32>
;CHECK: sub nsw <4 x i32>
;CHECK: select <4 x i1>
;CHECK: store <4 x i32>
;CHECK: ret
define void @example1(i32* nocapture %a, i32 %n, i32 %wsize) nounwind uwtable ssp {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %do.body ]
  %p.0 = phi i32* [ %a, %entry ], [ %incdec.ptr, %do.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.0, i64 -1
  %0 = load i32, i32* %incdec.ptr, align 4
  %cmp = icmp slt i32 %0, %wsize
  %sub = sub nsw i32 %0, %wsize
  %cond = select i1 %cmp, i32 0, i32 %sub
  store i32 %cond, i32* %incdec.ptr, align 4
  %dec = add nsw i32 %n.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret void
}
