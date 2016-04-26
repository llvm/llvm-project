; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx -fixup-byte-word-insts=1 < %s | FileCheck -check-prefix=CHECK -check-prefix=BWON %s
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx -fixup-byte-word-insts=0 < %s | FileCheck -check-prefix=CHECK -check-prefix=BWOFF %s
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx -addr-sink-using-gep=1 < %s | FileCheck -check-prefix=CHECK -check-prefix=BWON %s

%struct.A = type { i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.B = type { i32, i32, i32, i32, i32, i32, i32, i32 }

; CHECK-LABEL: merge_const_store:
; save 1,2,3 ... as one big integer.
; CHECK: movabsq $578437695752307201
; CHECK: ret
define void @merge_const_store(i32 %count, %struct.A* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.A* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %2, align 1
  %3 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %3, align 1
  %4 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %4, align 1
  %5 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %5, align 1
  %6 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 4
  store i8 5, i8* %6, align 1
  %7 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %8, align 1
  %9 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %9, align 1
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}

; No vectors because we use noimplicitfloat
; CHECK-LABEL: merge_const_store_no_vec:
; CHECK-NOT: vmovups
; CHECK: ret
define void @merge_const_store_no_vec(i32 %count, %struct.B* nocapture %p) noimplicitfloat{
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.B* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 0
  store i32 0, i32* %2, align 4
  %3 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 1
  store i32 0, i32* %3, align 4
  %4 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 2
  store i32 0, i32* %4, align 4
  %5 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 3
  store i32 0, i32* %5, align 4
  %6 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 4
  store i32 0, i32* %6, align 4
  %7 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 5
  store i32 0, i32* %7, align 4
  %8 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 6
  store i32 0, i32* %8, align 4
  %9 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 7
  store i32 0, i32* %9, align 4
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}

; Move the constants using a single vector store.
; CHECK-LABEL: merge_const_store_vec:
; CHECK: vmovups
; CHECK: ret
define void @merge_const_store_vec(i32 %count, %struct.B* nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge
.lr.ph:
  %i.02 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %.01 = phi %struct.B* [ %11, %.lr.ph ], [ %p, %0 ]
  %2 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 0
  store i32 0, i32* %2, align 4
  %3 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 1
  store i32 0, i32* %3, align 4
  %4 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 2
  store i32 0, i32* %4, align 4
  %5 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 3
  store i32 0, i32* %5, align 4
  %6 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 4
  store i32 0, i32* %6, align 4
  %7 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 5
  store i32 0, i32* %7, align 4
  %8 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 6
  store i32 0, i32* %8, align 4
  %9 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 7
  store i32 0, i32* %9, align 4
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}

; Move the first 4 constants as a single vector. Move the rest as scalars.
; CHECK-LABEL: merge_nonconst_store:
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
  %2 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 0
  store i8 1, i8* %2, align 1
  %3 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 1
  store i8 2, i8* %3, align 1
  %4 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 2
  store i8 3, i8* %4, align 1
  %5 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 3
  store i8 4, i8* %5, align 1
  %6 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 4
  store i8 %zz, i8* %6, align 1                     ;  <----------- Not a const;
  %7 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 5
  store i8 6, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 6
  store i8 7, i8* %8, align 1
  %9 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 7
  store i8 8, i8* %9, align 1
  %10 = add nsw i32 %i.02, 1
  %11 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %10, %count
  br i1 %exitcond, label %._crit_edge, label %.lr.ph
._crit_edge:
  ret void
}


; CHECK-LABEL: merge_loads_i16:
;  load:
; BWON:  movzwl
; BWOFF: movw
;  store:
; CHECK: movw
; CHECK: ret
define void @merge_loads_i16(i32 %count, %struct.A* noalias nocapture %q, %struct.A* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.A, %struct.A* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.A, %struct.A* %q, i64 0, i32 1
  br label %4

; <label>:4                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %9, %4 ]
  %.01 = phi %struct.A* [ %p, %.lr.ph ], [ %10, %4 ]
  %5 = load i8, i8* %2, align 1
  %6 = load i8, i8* %3, align 1
  %7 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 0
  store i8 %5, i8* %7, align 1
  %8 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 0, i32 1
  store i8 %6, i8* %8, align 1
  %9 = add nsw i32 %i.02, 1
  %10 = getelementptr inbounds %struct.A, %struct.A* %.01, i64 1
  %exitcond = icmp eq i32 %9, %count
  br i1 %exitcond, label %._crit_edge, label %4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

; The loads and the stores are interleaved. Can't merge them.
; CHECK-LABEL: no_merge_loads:
; BWON:  movzbl
; BWOFF: movb
; CHECK: movb
; BWON:  movzbl
; BWOFF: movb
; CHECK: movb
; CHECK: ret
define void @no_merge_loads(i32 %count, %struct.A* noalias nocapture %q, %struct.A* noalias nocapture %p) nounwind uwtable noinline ssp {
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


; CHECK-LABEL: merge_loads_integer:
;  load:
; CHECK: movq
;  store:
; CHECK: movq
; CHECK: ret
define void @merge_loads_integer(i32 %count, %struct.B* noalias nocapture %q, %struct.B* noalias nocapture %p) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 0
  %3 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 1
  br label %4

; <label>:4                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %9, %4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %10, %4 ]
  %5 = load i32, i32* %2
  %6 = load i32, i32* %3
  %7 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 0
  store i32 %5, i32* %7
  %8 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 1
  store i32 %6, i32* %8
  %9 = add nsw i32 %i.02, 1
  %10 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %9, %count
  br i1 %exitcond, label %._crit_edge, label %4

._crit_edge:                                      ; preds = %4, %0
  ret void
}


; CHECK-LABEL: merge_loads_vector:
;  load:
; CHECK: movups
;  store:
; CHECK: movups
; CHECK: ret
define void @merge_loads_vector(i32 %count, %struct.B* noalias nocapture %q, %struct.B* noalias nocapture %p) nounwind uwtable noinline ssp {
  %a1 = icmp sgt i32 %count, 0
  br i1 %a1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %a2 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 0
  %a3 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 1
  %a4 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 2
  %a5 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 3
  br label %block4

block4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %c9, %block4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %c10, %block4 ]
  %a7 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 0
  %a8 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 1
  %a9 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 2
  %a10 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 3
  %b1 = load i32, i32* %a2
  %b2 = load i32, i32* %a3
  %b3 = load i32, i32* %a4
  %b4 = load i32, i32* %a5
  store i32 %b1, i32* %a7
  store i32 %b2, i32* %a8
  store i32 %b3, i32* %a9
  store i32 %b4, i32* %a10
  %c9 = add nsw i32 %i.02, 1
  %c10 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %c9, %count
  br i1 %exitcond, label %._crit_edge, label %block4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

;; On x86, even unaligned copies should be merged to vector ops.
;; TODO: however, this cannot happen at the moment, due to brokenness
;; in MergeConsecutiveStores. See UseAA FIXME in DAGCombiner.cpp
;; visitSTORE.

; CHECK-LABEL: merge_loads_no_align:
;  load:
; CHECK-NOT: vmovups ;; TODO
;  store:
; CHECK-NOT: vmovups ;; TODO
; CHECK: ret
define void @merge_loads_no_align(i32 %count, %struct.B* noalias nocapture %q, %struct.B* noalias nocapture %p) nounwind uwtable noinline ssp {
  %a1 = icmp sgt i32 %count, 0
  br i1 %a1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %a2 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 0
  %a3 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 1
  %a4 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 2
  %a5 = getelementptr inbounds %struct.B, %struct.B* %q, i64 0, i32 3
  br label %block4

block4:                                       ; preds = %4, %.lr.ph
  %i.02 = phi i32 [ 0, %.lr.ph ], [ %c9, %block4 ]
  %.01 = phi %struct.B* [ %p, %.lr.ph ], [ %c10, %block4 ]
  %a7 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 0
  %a8 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 1
  %a9 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 2
  %a10 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 0, i32 3
  %b1 = load i32, i32* %a2, align 1
  %b2 = load i32, i32* %a3, align 1
  %b3 = load i32, i32* %a4, align 1
  %b4 = load i32, i32* %a5, align 1
  store i32 %b1, i32* %a7, align 1
  store i32 %b2, i32* %a8, align 1
  store i32 %b3, i32* %a9, align 1
  store i32 %b4, i32* %a10, align 1
  %c9 = add nsw i32 %i.02, 1
  %c10 = getelementptr inbounds %struct.B, %struct.B* %.01, i64 1
  %exitcond = icmp eq i32 %c9, %count
  br i1 %exitcond, label %._crit_edge, label %block4

._crit_edge:                                      ; preds = %4, %0
  ret void
}

; Make sure that we merge the consecutive load/store sequence below and use a
; word (16 bit) instead of a byte copy.
; CHECK-LABEL: MergeLoadStoreBaseIndexOffset:
; BWON: movzwl   (%{{.*}},%{{.*}}), %e[[REG:[a-z]+]]
; BWOFF: movw    (%{{.*}},%{{.*}}), %[[REG:[a-z]+]]
; CHECK: movw    %[[REG]], (%{{.*}})
define void @MergeLoadStoreBaseIndexOffset(i64* %a, i8* %b, i8* %c, i32 %n) {
  br label %1

; <label>:1
  %.09 = phi i32 [ %n, %0 ], [ %11, %1 ]
  %.08 = phi i8* [ %b, %0 ], [ %10, %1 ]
  %.0 = phi i64* [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i64, i64* %.0, i64 1
  %3 = load i64, i64* %.0, align 1
  %4 = getelementptr inbounds i8, i8* %c, i64 %3
  %5 = load i8, i8* %4, align 1
  %6 = add i64 %3, 1
  %7 = getelementptr inbounds i8, i8* %c, i64 %6
  %8 = load i8, i8* %7, align 1
  store i8 %5, i8* %.08, align 1
  %9 = getelementptr inbounds i8, i8* %.08, i64 1
  store i8 %8, i8* %9, align 1
  %10 = getelementptr inbounds i8, i8* %.08, i64 2
  %11 = add nsw i32 %.09, -1
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %1

; <label>:13
  ret void
}

; Make sure that we merge the consecutive load/store sequence below and use a
; word (16 bit) instead of a byte copy even if there are intermediate sign
; extensions.
; CHECK-LABEL: MergeLoadStoreBaseIndexOffsetSext:
; BWON: movzwl   (%{{.*}},%{{.*}}), %e[[REG:[a-z]+]]
; BWOFF: movw    (%{{.*}},%{{.*}}), %[[REG:[a-z]+]]
; CHECK: movw    %[[REG]], (%{{.*}})
define void @MergeLoadStoreBaseIndexOffsetSext(i8* %a, i8* %b, i8* %c, i32 %n) {
  br label %1

; <label>:1
  %.09 = phi i32 [ %n, %0 ], [ %12, %1 ]
  %.08 = phi i8* [ %b, %0 ], [ %11, %1 ]
  %.0 = phi i8* [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i8, i8* %.0, i64 1
  %3 = load i8, i8* %.0, align 1
  %4 = sext i8 %3 to i64
  %5 = getelementptr inbounds i8, i8* %c, i64 %4
  %6 = load i8, i8* %5, align 1
  %7 = add i64 %4, 1
  %8 = getelementptr inbounds i8, i8* %c, i64 %7
  %9 = load i8, i8* %8, align 1
  store i8 %6, i8* %.08, align 1
  %10 = getelementptr inbounds i8, i8* %.08, i64 1
  store i8 %9, i8* %10, align 1
  %11 = getelementptr inbounds i8, i8* %.08, i64 2
  %12 = add nsw i32 %.09, -1
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %1

; <label>:14
  ret void
}

; However, we can only merge ignore sign extensions when they are on all memory
; computations;
; CHECK-LABEL: loadStoreBaseIndexOffsetSextNoSex:
; CHECK-NOT: movw    (%{{.*}},%{{.*}}), [[REG:%[a-z]+]]
; CHECK-NOT: movw    [[REG]], (%{{.*}})
define void @loadStoreBaseIndexOffsetSextNoSex(i8* %a, i8* %b, i8* %c, i32 %n) {
  br label %1

; <label>:1
  %.09 = phi i32 [ %n, %0 ], [ %12, %1 ]
  %.08 = phi i8* [ %b, %0 ], [ %11, %1 ]
  %.0 = phi i8* [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i8, i8* %.0, i64 1
  %3 = load i8, i8* %.0, align 1
  %4 = sext i8 %3 to i64
  %5 = getelementptr inbounds i8, i8* %c, i64 %4
  %6 = load i8, i8* %5, align 1
  %7 = add i8 %3, 1
  %wrap.4 = sext i8 %7 to i64
  %8 = getelementptr inbounds i8, i8* %c, i64 %wrap.4
  %9 = load i8, i8* %8, align 1
  store i8 %6, i8* %.08, align 1
  %10 = getelementptr inbounds i8, i8* %.08, i64 1
  store i8 %9, i8* %10, align 1
  %11 = getelementptr inbounds i8, i8* %.08, i64 2
  %12 = add nsw i32 %.09, -1
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %1

; <label>:14
  ret void
}

; PR21711 ( http://llvm.org/bugs/show_bug.cgi?id=21711 )
define void @merge_vec_element_store(<8 x float> %v, float* %ptr) {
  %vecext0 = extractelement <8 x float> %v, i32 0
  %vecext1 = extractelement <8 x float> %v, i32 1
  %vecext2 = extractelement <8 x float> %v, i32 2
  %vecext3 = extractelement <8 x float> %v, i32 3
  %vecext4 = extractelement <8 x float> %v, i32 4
  %vecext5 = extractelement <8 x float> %v, i32 5
  %vecext6 = extractelement <8 x float> %v, i32 6
  %vecext7 = extractelement <8 x float> %v, i32 7
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 1
  %arrayidx2 = getelementptr inbounds float, float* %ptr, i64 2
  %arrayidx3 = getelementptr inbounds float, float* %ptr, i64 3
  %arrayidx4 = getelementptr inbounds float, float* %ptr, i64 4
  %arrayidx5 = getelementptr inbounds float, float* %ptr, i64 5
  %arrayidx6 = getelementptr inbounds float, float* %ptr, i64 6
  %arrayidx7 = getelementptr inbounds float, float* %ptr, i64 7
  store float %vecext0, float* %ptr, align 4
  store float %vecext1, float* %arrayidx1, align 4
  store float %vecext2, float* %arrayidx2, align 4
  store float %vecext3, float* %arrayidx3, align 4
  store float %vecext4, float* %arrayidx4, align 4
  store float %vecext5, float* %arrayidx5, align 4
  store float %vecext6, float* %arrayidx6, align 4
  store float %vecext7, float* %arrayidx7, align 4
  ret void

; CHECK-LABEL: merge_vec_element_store
; CHECK: vmovups
; CHECK-NEXT: vzeroupper
; CHECK-NEXT: retq
}

; PR21711 - Merge vector stores into wider vector stores.
; These should be merged into 32-byte stores.
define void @merge_vec_extract_stores(<8 x float> %v1, <8 x float> %v2, <4 x float>* %ptr) {
  %idx0 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 3
  %idx1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 4
  %idx2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 5
  %idx3 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 6
  %shuffle0 = shufflevector <8 x float> %v1, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %shuffle1 = shufflevector <8 x float> %v1, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle2 = shufflevector <8 x float> %v2, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %shuffle3 = shufflevector <8 x float> %v2, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  store <4 x float> %shuffle0, <4 x float>* %idx0, align 16
  store <4 x float> %shuffle1, <4 x float>* %idx1, align 16
  store <4 x float> %shuffle2, <4 x float>* %idx2, align 16
  store <4 x float> %shuffle3, <4 x float>* %idx3, align 16
  ret void

; CHECK-LABEL: merge_vec_extract_stores
; CHECK:      vmovups %ymm0, 48(%rdi)
; CHECK-NEXT: vmovups %ymm1, 80(%rdi)
; CHECK-NEXT: vzeroupper
; CHECK-NEXT: retq
}

; Merging vector stores when sourced from vector loads is not currently handled.
define void @merge_vec_stores_from_loads(<4 x float>* %v, <4 x float>* %ptr) {
  %load_idx0 = getelementptr inbounds <4 x float>, <4 x float>* %v, i64 0
  %load_idx1 = getelementptr inbounds <4 x float>, <4 x float>* %v, i64 1
  %v0 = load <4 x float>, <4 x float>* %load_idx0
  %v1 = load <4 x float>, <4 x float>* %load_idx1
  %store_idx0 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 0
  %store_idx1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 1
  store <4 x float> %v0, <4 x float>* %store_idx0, align 16
  store <4 x float> %v1, <4 x float>* %store_idx1, align 16
  ret void

; CHECK-LABEL: merge_vec_stores_from_loads
; CHECK:      vmovaps
; CHECK-NEXT: vmovaps
; CHECK-NEXT: vmovaps
; CHECK-NEXT: vmovaps
; CHECK-NEXT: retq
}

; Merging vector stores when sourced from a constant vector is not currently handled. 
define void @merge_vec_stores_of_constants(<4 x i32>* %ptr) {
  %idx0 = getelementptr inbounds <4 x i32>, <4 x i32>* %ptr, i64 3
  %idx1 = getelementptr inbounds <4 x i32>, <4 x i32>* %ptr, i64 4
  store <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32>* %idx0, align 16
  store <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32>* %idx1, align 16
  ret void

; CHECK-LABEL: merge_vec_stores_of_constants
; CHECK:      vxorps
; CHECK-NEXT: vmovaps
; CHECK-NEXT: vmovaps
; CHECK-NEXT: retq
}

; This is a minimized test based on real code that was failing.
; We could merge stores (and loads) like this...

define void @merge_vec_element_and_scalar_load([6 x i64]* %array) {
  %idx0 = getelementptr inbounds [6 x i64], [6 x i64]* %array, i64 0, i64 0
  %idx1 = getelementptr inbounds [6 x i64], [6 x i64]* %array, i64 0, i64 1
  %idx4 = getelementptr inbounds [6 x i64], [6 x i64]* %array, i64 0, i64 4
  %idx5 = getelementptr inbounds [6 x i64], [6 x i64]* %array, i64 0, i64 5

  %a0 = load i64, i64* %idx0, align 8
  store i64 %a0, i64* %idx4, align 8

  %b = bitcast i64* %idx1 to <2 x i64>*
  %v = load <2 x i64>, <2 x i64>* %b, align 8
  %a1 = extractelement <2 x i64> %v, i32 0
  store i64 %a1, i64* %idx5, align 8
  ret void

; CHECK-LABEL: merge_vec_element_and_scalar_load
; CHECK:      movq	(%rdi), %rax
; CHECK-NEXT: movq	%rax, 32(%rdi)
; CHECK-NEXT: movq	8(%rdi), %rax
; CHECK-NEXT: movq	%rax, 40(%rdi)
; CHECK-NEXT: retq
}
