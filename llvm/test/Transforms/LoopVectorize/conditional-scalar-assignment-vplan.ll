; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-vector-width=4 -disable-output 2>&1 < %s | FileCheck %s

; REQUIRES: asserts

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
; CHECK: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.*]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.*]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VECTC:%.*]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<[[ORIGTC:%.*]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[CIV:%.*]]> = CANONICAL-INDUCTION ir<0>, vp<[[INDEXNEXT:%.*]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[DATAPHI:%.*]]> = phi ir<-1>, vp<[[DATASELECT:%.*]]>
; CHECK-NEXT:     WIDEN-PHI vp<[[MASKPHI:%.*]]> = phi [ ir<false>, vector.ph ], [ vp<[[MASKSELECT:%.*]]>, vector.body ]
; CHECK-NEXT:     vp<[[STEPS:%.*]]> = SCALAR-STEPS vp<[[CIV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:     CLONE ir<[[LDADDR:%.*]]> = getelementptr inbounds ir<%data>, vp<[[STEPS:%.*]]>
; CHECK-NEXT:     vp<[[VPTR:%.*]]> = vector-pointer inbounds ir<[[LDADDR]]>
; CHECK-NEXT:     WIDEN ir<[[LD:%.*]]> = load vp<[[VPTR]]>
; CHECK-NEXT:     WIDEN ir<[[SELECTCMP:%.*]]> = icmp slt ir<%a>, ir<[[LD]]>
; CHECK-NEXT:     EMIT vp<[[ANYOF:%.*]]> = any-of ir<[[SELECTCMP]]>
; CHECK-NEXT:     EMIT vp<[[MASKSELECT]]> = select vp<[[ANYOF]]>, ir<[[SELECTCMP]]>, vp<[[MASKPHI]]>
; CHECK-NEXT:     EMIT vp<[[DATASELECT]]> = select vp<[[ANYOF]]>, ir<[[LD]]>, ir<[[DATAPHI]]>
; CHECK-NEXT:     EMIT vp<[[INDEXNEXT]]> = add nuw vp<[[CIV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[INDEXNEXT]]>, vp<[[VECTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[EXTRACTLAST:%.*]]> = extract-last-active vp<[[DATASELECT]]>, vp<[[MASKSELECT]]>, ir<-1>
; CHECK-NEXT:   EMIT vp<[[TCCMP:%.*]]> = icmp eq ir<[[ORIGTC]]>, vp<[[VECTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[TCCMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   [[SELECTLCSSA:%.*]] = phi i32 [ [[SELECTDATA:%.*]], %loop ] (extra operand: vp<[[EXTRACTLAST]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<[[RESUMEVAL:%.*]]> = phi [ vp<[[VECTC]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<[[MERGERDX:%.*]]> = phi [ vp<[[EXTRACTLAST]]>, middle.block ], [ ir<-1>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<loop>
;
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
