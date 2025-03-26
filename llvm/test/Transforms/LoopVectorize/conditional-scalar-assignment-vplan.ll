; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN:   -scalable-vectorization=on -force-target-supports-scalable-vectors \
; RUN:   -disable-output 2>&1 < %s | FileCheck %s


; This function is derived from the following C program:
; int simple_csa_int_select(int N, int *data, int a) {
;   int t = -1;
;   for (int i = 0; i < N; i++) {
;     if (a < data[i])
;       t = data[i];
;   }
;   return t;
; }
define i32 @simple_csa_int_select(i64 %N, ptr %data, i32 %a) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %data.phi = phi i32 [ -1, %entry ], [ %select.data, %loop ]
  %ld.addr = getelementptr inbounds i32, ptr %data, i64 %iv
  %ld = load i32, ptr %ld.addr, align 4
  %select.cmp = icmp slt i32 %a, %ld
  %select.data = select i1 %select.cmp, i32 %ld, i32 %data.phi
  %iv.next = add nuw nsw i64 %iv, 1
  %exit.cmp = icmp eq i64 %iv.next, %N
  br i1 %exit.cmp, label %exit, label %loop

exit:
  ret i32 %select.data
}


; CHECK: VPlan 'Initial VPlan for VF={vscale x 1},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = VF
; CHECK-NEXT: Live-in vp<%1> = VF * UF
; CHECK-NEXT: Live-in vp<%2> = vector-trip-count
; CHECK-NEXT: Live-in ir<%N> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<%data.phi> = phi ir<-1>, vp<%10>
; CHECK-NEXT:     ACTIVE-LANE-MASK-PHI vp<%4> = phi ir<false>, vp<%9>
; CHECK-NEXT:     vp<%5> = SCALAR-STEPS vp<%3>, ir<1>, vp<%0>
; CHECK-NEXT:     CLONE ir<%ld.addr> = getelementptr inbounds ir<%data>, vp<%5>
; CHECK-NEXT:     vp<%6> = vector-pointer ir<%ld.addr>
; CHECK-NEXT:     WIDEN ir<%ld> = load vp<%6>
; CHECK-NEXT:     WIDEN ir<%select.cmp> = icmp slt ir<%a>, ir<%ld>
; CHECK-NEXT:     EMIT vp<%7> = any-of ir<%select.cmp>
; CHECK-NEXT:     EMIT vp<%8> = broadcast vp<%7>
; CHECK-NEXT:     EMIT vp<%9> = select vp<%8>, ir<%select.cmp>, vp<%4>
; CHECK-NEXT:     EMIT vp<%10> = select vp<%8>, ir<%ld>, ir<%data.phi>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%3>, vp<%1>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<%12> = extract-last-active vp<%10>, vp<%9>, ir<-1>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq ir<%N>, vp<%2>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %select.data.lcssa = phi i32 [ %select.data, %loop ] (extra operand: vp<%12> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%2>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%12>, middle.block ], [ ir<-1>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:   IR   %data.phi = phi i32 [ -1, %entry ], [ %select.data, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:   IR   %ld.addr = getelementptr inbounds i32, ptr %data, i64 %iv
; CHECK-NEXT:   IR   %ld = load i32, ptr %ld.addr, align 4
; CHECK-NEXT:   IR   %select.cmp = icmp slt i32 %a, %ld
; CHECK-NEXT:   IR   %select.data = select i1 %select.cmp, i32 %ld, i32 %data.phi
; CHECK-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   IR   %exit.cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT: No successors
; CHECK-NEXT: }

; CHECK: Cost of 1 for VF vscale x 1: induction instruction   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT: Cost of 1 for VF vscale x 1: induction instruction   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT: Cost of 1 for VF vscale x 1: exit condition instruction   %exit.cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN-REDUCTION-PHI ir<%data.phi> = phi ir<-1>, vp<%10>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: ACTIVE-LANE-MASK-PHI vp<%4> = phi ir<false>, vp<%9>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: vp<%5> = SCALAR-STEPS vp<%3>, ir<1>, vp<%0>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: CLONE ir<%ld.addr> = getelementptr inbounds ir<%data>, vp<%5>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: vp<%6> = vector-pointer ir<%ld.addr>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%ld> = load vp<%6>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: WIDEN ir<%select.cmp> = icmp slt ir<%a>, ir<%ld>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: EMIT vp<%7> = any-of ir<%select.cmp>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%8> = broadcast vp<%7>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: EMIT vp<%9> = select vp<%8>, ir<%select.cmp>, vp<%4>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: EMIT vp<%10> = select vp<%8>, ir<%ld>, ir<%data.phi>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%index.next> = add nuw vp<%3>, vp<%1>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK-NEXT: Cost of 1 for VF vscale x 1: vector loop backedge
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%2>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%12>, middle.block ], [ ir<-1>, ir-bb<entry> ]
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %data.phi = phi i32 [ -1, %entry ], [ %select.data, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %ld.addr = getelementptr inbounds i32, ptr %data, i64 %iv
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %ld = load i32, ptr %ld.addr, align 4
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %select.cmp = icmp slt i32 %a, %ld
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %select.data = select i1 %select.cmp, i32 %ld, i32 %data.phi
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %exit.cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT: Cost of 1 for VF vscale x 1: EMIT vp<%12> = extract-last-active vp<%10>, vp<%9>, ir<-1>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT vp<%cmp.n> = icmp eq ir<%N>, vp<%2>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Cost of 0 for VF vscale x 1: IR   %select.data.lcssa = phi i32 [ %select.data, %loop ] (extra operand: vp<%12> from middle.block)
