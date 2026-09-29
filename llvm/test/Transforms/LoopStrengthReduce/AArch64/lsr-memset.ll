; RUN: llc < %s -O3 -mtriple=arm64-unknown-unknown -mcpu=cyclone -pre-RA-sched=list-hybrid | FileCheck %s
; <rdar://problem/11635990> [arm64] [lsr] Inefficient EA/loop-exit calc in bzero_phys
;
; LSR on loop %while.cond should reassociate non-address mode
; expressions at use %cmp16 to avoid sinking computation into %while.body18.
;
; Remove the -pre-RA-sched=list-hybrid option after fixing:
; <rdar://problem/12702735> [ARM64][coalescer] need better register
; coalescing for simple unit tests.

; CHECK: @memset
; CHECK: %while.body18{{$}}
; CHECK: str x{{[0-9]+}}, [x{{[0-9]+}}], #8
; First set the IVREG variable, then use it
; CHECK-NEXT: sub [[IVREG:x[0-9]+]],
; CHECK: [[IVREG]], #8
; CHECK-NEXT: cmp  [[IVREG]], #7
; CHECK-NEXT: b.hi
define ptr @memset(ptr %dest, i32 %val, i64 %len) nounwind ssp noimplicitfloat {
entry:
  %cmp = icmp eq i64 %len, 0
  br i1 %cmp, label %done, label %while.cond.preheader

while.cond.preheader:                             ; preds = %entry
  %conv = trunc i32 %val to i8
  br label %while.cond

while.cond:                                       ; preds = %while.body, %while.cond.preheader
  %ptr.0 = phi ptr [ %incdec.ptr, %while.body ], [ %dest, %while.cond.preheader ]
  %len.addr.0 = phi i64 [ %dec, %while.body ], [ %len, %while.cond.preheader ]
  %cond = icmp eq i64 %len.addr.0, 0
  br i1 %cond, label %done, label %land.rhs

land.rhs:                                         ; preds = %while.cond
  %0 = ptrtoint ptr %ptr.0 to i64
  %and = and i64 %0, 7
  %cmp5 = icmp eq i64 %and, 0
  br i1 %cmp5, label %if.end9, label %while.body

while.body:                                       ; preds = %land.rhs
  %incdec.ptr = getelementptr inbounds i8, ptr %ptr.0, i64 1
  store i8 %conv, ptr %ptr.0, align 1, !tbaa !0
  %dec = add i64 %len.addr.0, -1
  br label %while.cond

if.end9:                                          ; preds = %land.rhs
  %conv.mask = and i32 %val, 255
  %1 = zext i32 %conv.mask to i64
  %2 = shl nuw nsw i64 %1, 8
  %ins18 = or i64 %2, %1
  %3 = shl nuw nsw i64 %1, 16
  %ins15 = or i64 %ins18, %3
  %4 = shl nuw nsw i64 %1, 24
  %5 = shl nuw nsw i64 %1, 32
  %mask8 = or i64 %ins15, %4
  %6 = shl nuw nsw i64 %1, 40
  %mask5 = or i64 %mask8, %5
  %7 = shl nuw nsw i64 %1, 48
  %8 = shl nuw i64 %1, 56
  %mask2.masked = or i64 %mask5, %6
  %mask = or i64 %mask2.masked, %7
  %ins = or i64 %mask, %8
  %cmp1636 = icmp ugt i64 %len.addr.0, 7
  br i1 %cmp1636, label %while.body18, label %while.body29.lr.ph

while.body18:                                     ; preds = %if.end9, %while.body18
  %wideptr.038 = phi ptr [ %incdec.ptr19, %while.body18 ], [ %ptr.0, %if.end9 ]
  %len.addr.137 = phi i64 [ %sub, %while.body18 ], [ %len.addr.0, %if.end9 ]
  %incdec.ptr19 = getelementptr inbounds i64, ptr %wideptr.038, i64 1
  store i64 %ins, ptr %wideptr.038, align 8, !tbaa !2
  %sub = add i64 %len.addr.137, -8
  %cmp16 = icmp ugt i64 %sub, 7
  br i1 %cmp16, label %while.body18, label %while.end20

while.end20:                                      ; preds = %while.body18
  %cmp21 = icmp eq i64 %sub, 0
  br i1 %cmp21, label %done, label %while.body29.lr.ph

while.body29.lr.ph:                               ; preds = %while.end20, %if.end9
  %len.addr.1.lcssa49 = phi i64 [ %sub, %while.end20 ], [ %len.addr.0, %if.end9 ]
  %wideptr.0.lcssa48 = phi ptr [ %incdec.ptr19, %while.end20 ], [ %ptr.0, %if.end9 ]
  br label %while.body29

while.body29:                                     ; preds = %while.body29, %while.body29.lr.ph
  %len.addr.235 = phi i64 [ %len.addr.1.lcssa49, %while.body29.lr.ph ], [ %dec26, %while.body29 ]
  %ptr.134 = phi ptr [ %wideptr.0.lcssa48, %while.body29.lr.ph ], [ %incdec.ptr31, %while.body29 ]
  %dec26 = add i64 %len.addr.235, -1
  %incdec.ptr31 = getelementptr inbounds i8, ptr %ptr.134, i64 1
  store i8 %conv, ptr %ptr.134, align 1, !tbaa !0
  %cmp27 = icmp eq i64 %dec26, 0
  br i1 %cmp27, label %done, label %while.body29

done:                                             ; preds = %while.cond, %while.body29, %while.end20, %entry
  ret ptr %dest
}

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"long long", !0}
