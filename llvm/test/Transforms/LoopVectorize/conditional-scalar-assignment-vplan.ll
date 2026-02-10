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

; This function is derived from the following C program:
; int simple_csa_int_load(int* a, int* b, int default_val, int N, int threshold) {
;   int result = default_val;
;   for (int i = 0; i < N; ++i)
;     if (a[i] > threshold)
;       result = b[i];
;   return result;
; }
define i32 @simple_csa_int_load(ptr noalias %a, ptr noalias %b, i32 %default_val, i64 %N, i32 %threshold) {
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
; CHECK-NEXT:     EMIT vp<[[CIV:%.*]]> = CANONICAL-INDUCTION ir<0>, vp<[[INDEXNEXT]]>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<[[DATAPHI:%.*]]> = phi ir<%default_val>, vp<[[DATASELECT:%.*]]>
; CHECK-NEXT:     WIDEN-PHI vp<[[MASKPHI:%.*]]> = phi [ ir<false>, vector.ph ], [ vp<[[MASKSELECT:%.*]]>, if.then.0 ]
; CHECK-NEXT:     vp<[[STEPS_A:%.*]]> = SCALAR-STEPS vp<[[CIV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:     CLONE ir<[[A_ADDR:%.*]]> = getelementptr inbounds nuw ir<%a>, vp<[[STEPS_A]]>
; CHECK-NEXT:     vp<[[A_VPTR:%.*]]> = vector-pointer inbounds nuw ir<[[A_ADDR]]>
; CHECK-NEXT:     WIDEN ir<[[LD_A:%.*]]> = load vp<[[A_VPTR]]>
; CHECK-NEXT:     WIDEN ir<[[IF_COND:%.*]]> = icmp sgt ir<[[LD_A]]>, ir<%threshold>
; CHECK-NEXT:   Successor(s): pred.load
; CHECK-EMPTY:
; CHECK-NEXT:  <xVFxUF> pred.load: {
; CHECK-NEXT:    pred.load.entry:
; CHECK-NEXT:      BRANCH-ON-MASK ir<[[IF_COND]]>
; CHECK-NEXT:    Successor(s): pred.load.if, pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.load.if:
; CHECK-NEXT:      vp<[[STEPS_B:%.*]]> = SCALAR-STEPS vp<[[CIV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:      REPLICATE ir<[[B_ADDR:%.*]]> = getelementptr inbounds nuw ir<%b>, vp<[[STEPS_B]]>
; CHECK-NEXT:      REPLICATE ir<[[LD_B:%.*]]> = load ir<[[B_ADDR]]> (S->V)
; CHECK-NEXT:    Successor(s): pred.load.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.load.continue:
; CHECK-NEXT:      PHI-PREDICATED-INSTRUCTION vp<[[B_VEC:%.*]]> = ir<[[LD_B]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): if.then.0
; CHECK-EMPTY:
; CHECK-NEXT:   if.then.0:
; CHECK-NEXT:     EMIT vp<[[ANYOF:%.*]]> = any-of ir<[[IF_COND]]>
; CHECK-NEXT:     EMIT vp<[[MASKSELECT]]> = select vp<[[ANYOF]]>, ir<[[IF_COND]]>, vp<[[MASKPHI]]>
; CHECK-NEXT:     EMIT vp<[[DATASELECT]]> = select vp<[[ANYOF]]>, vp<[[B_VEC]]>, ir<[[DATAPHI]]>
; CHECK-NEXT:     EMIT vp<[[INDEXNEXT:%.*]]> = add nuw vp<[[CIV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[INDEXNEXT]]>, vp<[[VECTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[EXTRACTLAST:%.*]]> = extract-last-active vp<[[DATASELECT]]>, vp<[[MASKSELECT]]>, ir<%default_val>
; CHECK-NEXT:   EMIT vp<[[TCCMP:%.*]]> = icmp eq ir<[[ORIGTC]]>, vp<[[VECTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[TCCMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %select.data.lcssa = phi i32 [ %select.data, %latch ] (extra operand: vp<[[EXTRACTLAST]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<[[VECTC]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[EXTRACTLAST]]>, middle.block ], [ ir<%default_val>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<loop>
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %data.phi = phi i32 [ %default_val, %entry ], [ %select.data, %latch ]
  %a.addr = getelementptr inbounds nuw i32, ptr %a, i64 %iv
  %ld.a = load i32, ptr %a.addr, align 4
  %if.cond = icmp sgt i32 %ld.a, %threshold
  br i1 %if.cond, label %if.then, label %latch

if.then:
  %b.addr = getelementptr inbounds nuw i32, ptr %b, i64 %iv
  %ld.b = load i32, ptr %b.addr, align 4
  br label %latch

latch:
  %select.data = phi i32 [ %ld.b, %if.then ], [ %data.phi, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exit.cmp = icmp eq i64 %iv.next, %N
  br i1 %exit.cmp, label %exit, label %loop

exit:
  ret i32 %select.data
}
