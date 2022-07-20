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
                  %struct.A* noalias nocapture %q,
                  %struct.A* noalias nocapture %p)
                    nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A, %struct.A* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.A, %struct.A* %q, i64 0, i32 1
  br label %a4

a4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %a9, %a4 ]
  %.01 = phi %struct.A* [ %p, %.lr.ph ], [ %a10, %a4 ]
  %a5 = load i8, i8* %2, align 1
  %a7 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 0
  store i8 %a5, i8* %a7, align 1
  %a8 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 1
  %a6 = load i8, i8* %3, align 1
  store i8 %a6, i8* %a8, align 1
  %a9 = add nsw i32 %i.02, 1
  %a10 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %a9, %count
  br i1 %exitcond, label %._crit_edge, label %a4

._crit_edge:                                      ; preds = %4, %0
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
                  %struct.B* noalias nocapture %q,
                  %struct.B* noalias nocapture %p)
                    nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 1
  br label %a4

a4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %a9, %a4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %a10, %a4 ]
  %a5 = load i16, i16* %2, align 2
  %a7 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 0
  store i16 %a5, i16* %a7, align 2
  %a8 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 1
  %a6 = load i16, i16* %3, align 2
  store i16 %a6, i16* %a8, align 2
  %a9 = add nsw i32 %i.02, 1
  %a10 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %a9, %count
  br i1 %exitcond, label %._crit_edge, label %a4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

; This test contains nothing but a simple byte load and store.
; movb encodes smaller, but we use movzbl for the load for better perf.
; CHECK-LABEL: foo3:
; BWON:  movzbl
; BWOFF: movb
; CHECK: movb
define void @foo3(i8 *%dst, i8 *%src) {
  %t0 = load i8, i8 *%src, align 1
  store i8 %t0, i8 *%dst, align 1
  ret void
}

; This test contains nothing but a simple word load and store.  Since
; movw and movzwl are the same size, we should always choose to use
; movzwl instead.
; CHECK-LABEL: foo4:
; BWON:  movzwl
; BWOFF: movw
; CHECK: movw
define void @foo4(i16 *%dst, i16 *%src) {
  %t0 = load i16, i16 *%src, align 2
  store i16 %t0, i16 *%dst, align 2
  ret void
}
