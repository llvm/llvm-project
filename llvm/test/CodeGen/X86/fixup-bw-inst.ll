; RUN: llc -fixup-byte-word-insts=1 < %s | \
; RUN: FileCheck -check-prefix CHECK -check-prefix BWON %s
; RUN: llc -fixup-byte-word-insts=0 < %s | \
; RUN: FileCheck -check-prefix CHECK -check-prefix BWOFF %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.A = type { i8, i8, i8, i8, i8, i8, i8, i8 }

; This has byte loads interspersed with byte stores, in a single
; basic-block loop.  The upper portion should be dead, so the movb loads
; should have been changed into movzbl instead.
; CHECK-LABEL: foo1
; load:
; BWON:  movzbl
; BWOFF: movb
; store:
; CHECK: movb
; load:
; BWON: movzbl
; BWOFF: movb
; store:
; CHECK: movb
; CHECK: ret
define void @foo1(i32 %count,
                  ptr noalias nocapture %q,
                  ptr noalias nocapture %p)
                    nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A, ptr %q, i64 0, i32 1
  br label %a4

a4:                                       ; preds = %3, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %a9, %a4 ]
  %.01 = phi ptr [ %p, %.lr.ph ], [ %a10, %a4 ]
  %a5 = load i8, ptr %q, align 1
  store i8 %a5, ptr %.01, align 1
  %a8 = getelementptr inbounds %struct.A, ptr %.01, i64 0, i32 1
  %a6 = load i8, ptr %2, align 1
  store i8 %a6, ptr %a8, align 1
  %a9 = add nsw i32 %i.02, 1
  %a10 = getelementptr inbounds %struct.A, ptr %.01, i64 1
  %exitcond = icmp eq i32 %a9, %count
  br i1 %exitcond, label %._crit_edge, label %a4

._crit_edge:                                      ; preds = %3, %0
  ret void
}

%struct.B = type { i16, i16, i16, i16, i16, i16, i16, i16 }

; This has word loads interspersed with word stores.
; The upper portion should be dead, so the movw loads should have
; been changed into movzwl instead.
; CHECK-LABEL: foo2
; load:
; BWON:  movzwl
; BWOFF: movw
; store:
; CHECK: movw
; load:
; BWON:  movzwl
; BWOFF: movw
; store:
; CHECK: movw
; CHECK: ret
define void @foo2(i32 %count,
                  ptr noalias nocapture %q,
                  ptr noalias nocapture %p)
                    nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.B, ptr %q, i64 0, i32 1
  br label %a4

a4:                                       ; preds = %3, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %a9, %a4 ]
  %.01 = phi ptr [ %p, %.lr.ph ], [ %a10, %a4 ]
  %a5 = load i16, ptr %q, align 2
  store i16 %a5, ptr %.01, align 2
  %a8 = getelementptr inbounds %struct.B, ptr %.01, i64 0, i32 1
  %a6 = load i16, ptr %2, align 2
  store i16 %a6, ptr %a8, align 2
  %a9 = add nsw i32 %i.02, 1
  %a10 = getelementptr inbounds %struct.B, ptr %.01, i64 1
  %exitcond = icmp eq i32 %a9, %count
  br i1 %exitcond, label %._crit_edge, label %a4

._crit_edge:                                      ; preds = %3, %0
  ret void
}

; This test contains nothing but a simple byte load and store.  Since
; movb encodes smaller, we do not want to use movzbl unless in a tight loop.
; So this test checks that movb is used.
; CHECK-LABEL: foo3:
; CHECK: movb
; CHECK: movb
define void @foo3(ptr%dst, ptr%src) {
  %t0 = load i8, ptr%src, align 1
  store i8 %t0, ptr%dst, align 1
  ret void
}

; This test contains nothing but a simple word load and store.  Since
; movw and movzwl are the same size, we should always choose to use
; movzwl instead.
; CHECK-LABEL: foo4:
; BWON:  movzwl
; BWOFF: movw
; CHECK: movw
define void @foo4(ptr%dst, ptr%src) {
  %t0 = load i16, ptr%src, align 2
  store i16 %t0, ptr%dst, align 2
  ret void
}
