; REQUIRES: asserts
; RUN: opt -mtriple=riscv64-unknown-linux -mattr=+v -passes=slp-vectorizer \
; RUN:   -debug-only=SLP -S -disable-output < %s 2>&1 | FileCheck %s

; The 8 scalar i8 loads below deinterleave into two 4-wide streams (interleave
; factor 2), so this TreeEntry has E->Scalars.size() == 8 (the full combined
; width of both streams) and E->getInterleaveFactor() == 2.
;
; The STLF hazard check must use the load's real byte width, which is just
; Scalars.size() * ElementSize = 8 bytes: the interleave factor must not be
; multiplied in again, since Scalars already lists every lane across all
; interleaved streams. Multiplying by the factor a second time doubles the
; assumed load width to 16 bytes, which is wide enough to make an otherwise
; strided-independent access (load base is 28 bytes behind the store, stride
; 16) look like a false forwarding conflict.
;
; CHECK: SLP: STLF check: VF=4 ElementSize=1 VectorStoreBytes=4
; CHECK-NEXT: SLP: STLF: load={{.*}}distance=28 bytes from chain base
; CHECK-NEXT: SLP: STLF: strided-independent (stride 16), no future re-read -> no conflict
; CHECK-NOT: SLP: Store-load forwarding conflict

define void @stlf_interleaved(ptr noalias %A, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 28, %entry ], [ %i.next, %loop ]
  %loadbase = sub i64 %i, 28
  %lm0 = add i64 %loadbase, 0
  %lm1 = add i64 %loadbase, 1
  %lm2 = add i64 %loadbase, 2
  %lm3 = add i64 %loadbase, 3
  %lm4 = add i64 %loadbase, 4
  %lm5 = add i64 %loadbase, 5
  %lm6 = add i64 %loadbase, 6
  %lm7 = add i64 %loadbase, 7
  %p0 = getelementptr inbounds i8, ptr %A, i64 %lm0
  %p1 = getelementptr inbounds i8, ptr %A, i64 %lm1
  %p2 = getelementptr inbounds i8, ptr %A, i64 %lm2
  %p3 = getelementptr inbounds i8, ptr %A, i64 %lm3
  %p4 = getelementptr inbounds i8, ptr %A, i64 %lm4
  %p5 = getelementptr inbounds i8, ptr %A, i64 %lm5
  %p6 = getelementptr inbounds i8, ptr %A, i64 %lm6
  %p7 = getelementptr inbounds i8, ptr %A, i64 %lm7
  %a0 = load i8, ptr %p0, align 1
  %a1 = load i8, ptr %p1, align 1
  %a2 = load i8, ptr %p2, align 1
  %a3 = load i8, ptr %p3, align 1
  %a4 = load i8, ptr %p4, align 1
  %a5 = load i8, ptr %p5, align 1
  %a6 = load i8, ptr %p6, align 1
  %a7 = load i8, ptr %p7, align 1
  %r0 = sub i8 %a0, %a1
  %r1 = sub i8 %a2, %a3
  %r2 = sub i8 %a4, %a5
  %r3 = sub i8 %a6, %a7
  %o0 = getelementptr inbounds i8, ptr %A, i64 %i
  %oi1 = add i64 %i, 1
  %o1 = getelementptr inbounds i8, ptr %A, i64 %oi1
  %oi2 = add i64 %i, 2
  %o2 = getelementptr inbounds i8, ptr %A, i64 %oi2
  %oi3 = add i64 %i, 3
  %o3 = getelementptr inbounds i8, ptr %A, i64 %oi3
  store i8 %r0, ptr %o0, align 1
  store i8 %r1, ptr %o1, align 1
  store i8 %r2, ptr %o2, align 1
  store i8 %r3, ptr %o3, align 1
  %i.next = add nuw nsw i64 %i, 16
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
