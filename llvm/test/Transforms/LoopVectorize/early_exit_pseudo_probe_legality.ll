; REQUIRES: asserts
;; Stores in uncountable early exit loops are only vectorized with
;; -enable-early-exit-vectorization-with-side-effects; the flag does not affect
;; the read-only loops below.
; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize -enable-early-exit-vectorization-with-side-effects -force-vector-width=4 -disable-output 2>&1 | FileCheck %s

declare void @init_mem(ptr, i64);

;; A pseudo probe reports may-read and may-write because it is modelled as
;; accessing inaccessible memory, but it carries no real memory dependence, so
;; the loop is still read-only for the purposes of early exit vectorization.
define i64 @read_only_early_exit_with_pseudo_probe() {
; CHECK-LABEL: LV: Checking a loop in 'read_only_early_exit_with_pseudo_probe'
; CHECK:       LV: Found an early exit loop with symbolic max backedge taken count: 63
; CHECK-NEXT:  LV: We can vectorize this loop!
; CHECK-NOT:   LV: Not vectorizing:
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %retval = phi i64 [ 0, %loop ], [ 1, %loop.inc ]
  ret i64 %retval
}

;; Same loop as @loop_contains_store_condition_load_has_single_user in
;; early_exit_store_legality.ll, with a pseudo probe added. The probe must
;; neither trip the "complex writes to memory" check nor prevent the exit
;; condition load from being moved.
define void @store_early_exit_with_pseudo_probe(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'store_early_exit_with_pseudo_probe'
; CHECK:       LV: We can vectorize this loop!
; CHECK-NOT:   LV: Not vectorizing:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 2, i32 0, i64 -1)
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

;; llvm.sideeffect is modelled the same way as llvm.pseudoprobe, but it is not a
;; profiling placeholder, so it must still block vectorization.
define i64 @read_only_early_exit_with_sideeffect() {
; CHECK-LABEL: LV: Checking a loop in 'read_only_early_exit_with_sideeffect'
; CHECK:       LV: Not vectorizing: Complex writes to memory unsupported in early exit loops.
; CHECK-NOT:   LV: We can vectorize this loop!
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  call void @llvm.sideeffect()
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %retval = phi i64 [ 0, %loop ], [ 1, %loop.inc ]
  ret i64 %retval
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)
declare void @llvm.sideeffect()

!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 5116412291814990879, i64 52824598631, !"read_only_early_exit_with_pseudo_probe"}
