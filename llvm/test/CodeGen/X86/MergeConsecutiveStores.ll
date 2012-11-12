; RUN: llc -march=x86-64 -mcpu=corei7 -mattr=+avx < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.A = type { i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.B = type { i32, i32, i32, i32, i32, i32, i32, i32 }

; CHECK: merge_const_store
; save 1,2,3 ... as one big integer.
; CHECK: movabsq $578437695752307201
; CHECK: ret
define void @merge_const_store(i32 %count, %struct.A* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.A* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %2, align 1
  %3 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %3, align 1
  %4 = getelementptr inbounds %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %4, align 1
  %5 = getelementptr inbounds %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %5, align 1
  %6 = getelementptr inbounds %struct.A* %.01, i64 0, i32 4
  store i8 5, i8* %6, align 1
  %7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %8, align 1
  %9 = getelementptr inbounds %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %9, align 1
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}

; Move the constants using a single vector store.
; CHECK: merge_const_store_vec
; CHECK: vmovups  %ymm0, (%rsi)
; CHECK: ret
define void @merge_const_store_vec(i32 %count, %struct.B* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.B* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.B* %.01, i64 0, i32 0
  store i32 0, i32* %2, align 4
  %3 = getelementptr inbounds %struct.B* %.01, i64 0, i32 1
  store i32 0, i32* %3, align 4
  %4 = getelementptr inbounds %struct.B* %.01, i64 0, i32 2
  store i32 0, i32* %4, align 4
  %5 = getelementptr inbounds %struct.B* %.01, i64 0, i32 3
  store i32 0, i32* %5, align 4
  %6 = getelementptr inbounds %struct.B* %.01, i64 0, i32 4
  store i32 0, i32* %6, align 4
  %7 = getelementptr inbounds %struct.B* %.01, i64 0, i32 5
  store i32 0, i32* %7, align 4
  %8 = getelementptr inbounds %struct.B* %.01, i64 0, i32 6
  store i32 0, i32* %8, align 4
  %9 = getelementptr inbounds %struct.B* %.01, i64 0, i32 7
  store i32 0, i32* %9, align 4
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}

; Move the first 4 constants as a single vector. Move the rest as scalars.
; CHECK: merge_nonconst_store
; CHECK: movl $67305985
; CHECK: movb
; CHECK: movb
; CHECK: movb
; CHECK: movb
; CHECK: ret
define void @merge_nonconst_store(i32 %count, i8 %zz, %struct.A* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.A* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %2, align 1
  %3 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %3, align 1
  %4 = getelementptr inbounds %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %4, align 1
  %5 = getelementptr inbounds %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %5, align 1
  %6 = getelementptr inbounds %struct.A* %.01, i64 0, i32 4
  store i8 %zz, i8* %6, align 1                     ;  <----------- Not a const;
  %7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %8, align 1
  %9 = getelementptr inbounds %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %9, align 1
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}


;CHECK: merge_loads_i16
; load:
;CHECK: movw
; store:
;CHECK: movw
;CHECK: ret
define void @merge_loads_i16(i32 %count, %struct.A* noalias nocapture %q, %struct.A* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.A* %q, i64 0, i32 1
  br label %4

; <label>:4                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %9, %4 ]
  %.01 = phi %struct.A* [ %p, %.lr.ph ], [ %10, %4 ]
  %5 = load i8* %2, align 1
  %6 = load i8* %3, align 1
  %7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 %5, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  store i8 %6, i8* %8, align 1
  %9 = add nsw i32 %i.02, 1
  %10 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %9, %count
  br i1 %exitcond, label %._crit_edge, label %4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

; The loads and the stores are interleved. Can't merge them.
;CHECK: no_merge_loads
;CHECK: movb
;CHECK: movb
;CHECK: movb
;CHECK: movb
;CHECK: ret
define void @no_merge_loads(i32 %count, %struct.A* noalias nocapture %q, %struct.A* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.A* %q, i64 0, i32 1
  br label %a4

a4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %a9, %a4 ]
  %.01 = phi %struct.A* [ %p, %.lr.ph ], [ %a10, %a4 ]
  %a5 = load i8* %2, align 1
  %a7 = getelementptr inbounds %struct.A* %.01, i64 0, i32 0
  store i8 %a5, i8* %a7, align 1
  %a8 = getelementptr inbounds %struct.A* %.01, i64 0, i32 1
  %a6 = load i8* %3, align 1
  store i8 %a6, i8* %a8, align 1
  %a9 = add nsw i32 %i.02, 1
  %a10 = getelementptr inbounds %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %a9, %count
  br i1 %exitcond, label %._crit_edge, label %a4

._crit_edge:                                      ; preds = %4, %0
  ret void
}


;CHECK: merge_loads_integer
; load:
;CHECK: movq
; store:
;CHECK: movq
;CHECK: ret
define void @merge_loads_integer(i32 %count, %struct.B* noalias nocapture %q, %struct.B* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.B* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.B* %q, i64 0, i32 1
  br label %4

; <label>:4                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %9, %4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %10, %4 ]
  %5 = load i32* %2
  %6 = load i32* %3
  %7 = getelementptr inbounds %struct.B* %.01, i64 0, i32 0
  store i32 %5, i32* %7
  %8 = getelementptr inbounds %struct.B* %.01, i64 0, i32 1
  store i32 %6, i32* %8
  %9 = add nsw i32 %i.02, 1
  %10 = getelementptr inbounds %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %9, %count
  br i1 %exitcond, label %._crit_edge, label %4

._crit_edge:                                      ; preds = %4, %0
  ret void
}


;CHECK: merge_loads_vector
; load:
;CHECK: movups
; store:
;CHECK: movups
;CHECK: ret
define void @merge_loads_vector(i32 %count, %struct.B* noalias nocapture %q, %struct.B* noalias nocapture %p) nounwind uwtable noinline ssp {
  %a1 = icmp sgt i32 %count, 0
  br i1 %a1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %a2 = getelementptr inbounds %struct.B* %q, i64 0, i32 0
  %a3 = getelementptr inbounds %struct.B* %q, i64 0, i32 1
  %a4 = getelementptr inbounds %struct.B* %q, i64 0, i32 2
  %a5 = getelementptr inbounds %struct.B* %q, i64 0, i32 3
  br label %block4

block4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %c9, %block4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %c10, %block4 ]
  %a7 = getelementptr inbounds %struct.B* %.01, i64 0, i32 0
  %a8 = getelementptr inbounds %struct.B* %.01, i64 0, i32 1
  %a9 = getelementptr inbounds %struct.B* %.01, i64 0, i32 2
  %a10 = getelementptr inbounds %struct.B* %.01, i64 0, i32 3
  %b1 = load i32* %a2
  %b2 = load i32* %a3
  %b3 = load i32* %a4
  %b4 = load i32* %a5
  store i32 %b1, i32* %a7
  store i32 %b2, i32* %a8
  store i32 %b3, i32* %a9
  store i32 %b4, i32* %a10
  %c9 = add nsw i32 %i.02, 1
  %c10 = getelementptr inbounds %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %c9, %count
  br i1 %exitcond, label %._crit_edge, label %block4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

;CHECK: merge_loads_no_align
; load:
;CHECK: movl
;CHECK: movl
;CHECK: movl
;CHECK: movl
; store:
;CHECK: movl
;CHECK: movl
;CHECK: movl
;CHECK: movl
;CHECK: ret
define void @merge_loads_no_align(i32 %count, %struct.B* noalias nocapture %q, %struct.B* noalias nocapture %p) nounwind uwtable noinline ssp {
  %a1 = icmp sgt i32 %count, 0
  br i1 %a1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %a2 = getelementptr inbounds %struct.B* %q, i64 0, i32 0
  %a3 = getelementptr inbounds %struct.B* %q, i64 0, i32 1
  %a4 = getelementptr inbounds %struct.B* %q, i64 0, i32 2
  %a5 = getelementptr inbounds %struct.B* %q, i64 0, i32 3
  br label %block4

block4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %c9, %block4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %c10, %block4 ]
  %a7 = getelementptr inbounds %struct.B* %.01, i64 0, i32 0
  %a8 = getelementptr inbounds %struct.B* %.01, i64 0, i32 1
  %a9 = getelementptr inbounds %struct.B* %.01, i64 0, i32 2
  %a10 = getelementptr inbounds %struct.B* %.01, i64 0, i32 3
  %b1 = load i32* %a2, align 1
  %b2 = load i32* %a3, align 1
  %b3 = load i32* %a4, align 1
  %b4 = load i32* %a5, align 1
  store i32 %b1, i32* %a7, align 1
  store i32 %b2, i32* %a8, align 1
  store i32 %b3, i32* %a9, align 1
  store i32 %b4, i32* %a10, align 1
  %c9 = add nsw i32 %i.02, 1
  %c10 = getelementptr inbounds %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %c9, %count
  br i1 %exitcond, label %._crit_edge, label %block4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

