; This is the loop in c++ being vectorize in this file with
;vector.reverse
;  #pragma clang loop vectorize_width(4, scalable)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v \
; RUN: -debug-only=loop-vectorize -scalable-vectorization=on \
; RUN: -disable-output < %s 2>&1 | FileCheck %s

define void @vector_reverse_i64(ptr nocapture noundef writeonly %A, ptr nocapture noundef readonly %B, i32 noundef signext %n) {
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; CHECK-NEXT: Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: vp<[[OTC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT:   EMIT vp<[[OTC]]> = EXPAND SCEV (1 + (-1 * (1 umin %n))<nuw><nsw> + %n)
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<[[INDUCTION:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[INDEX_NEXT:%.+]]>
; CHECK-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%.+]]> = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; CHECK-NEXT:     EMIT-SCALAR vp<[[AVL:%.+]]> = phi [ vp<[[OTC]]>, vector.ph ], [ vp<[[AVL_NEXT:%.+]]>, vector.body ]
; CHECK-NEXT:     EMIT-SCALAR vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; CHECK-NEXT:     vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<%n> + vp<[[EVL_PHI]]> * ir<-1>
; CHECK-NEXT:     vp<[[SCALAR_STEPS:%.+]]> = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<-1>, vp<[[EVL]]>
; CHECK-NEXT:     CLONE ir<[[IDX:%.+]]> = add nsw vp<[[SCALAR_STEPS]]>, ir<-1>
; CHECK-NEXT:     CLONE ir<[[IDX_PROM:%.+]]> = zext ir<[[IDX]]>
; CHECK-NEXT:     CLONE ir<[[ARRAY_IDX_B:%.+]]> = getelementptr inbounds ir<[[B:%.+]]>, ir<[[IDX_PROM]]>
; CHECK-NEXT:     vp<[[VEC_END_PTR_B:%.+]]> = vector-end-pointer ir<[[ARRAY_IDX_B]]>, vp<[[EVL]]>
; CHECK-NEXT:     WIDEN ir<[[VAL_B:%.+]]> = vp.load vp<[[VEC_END_PTR_B]]>, vp<[[EVL]]>
; CHECK-NEXT:     WIDEN ir<[[ADD_RESULT:%.+]]> = add ir<[[VAL_B]]>, ir<1>
; CHECK-NEXT:     CLONE ir<[[ARRAY_IDX_A:%.+]]> = getelementptr inbounds ir<[[A:%.+]]>, ir<[[IDX_PROM]]>
; CHECK-NEXT:     vp<[[VEC_END_PTR_A:%.+]]> = vector-end-pointer ir<[[ARRAY_IDX_A]]>, vp<[[EVL]]>
; CHECK-NEXT:     WIDEN vp.store vp<[[VEC_END_PTR_A]]>, ir<[[ADD_RESULT]]>, vp<[[EVL]]>
; CHECK-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[EVL]]>, vp<[[EVL_PHI]]>
; CHECK-NEXT:     EMIT vp<[[AVL_NEXT]]> = sub nuw vp<[[AVL]]>, vp<[[EVL]]>
; CHECK-NEXT:     EMIT vp<[[INDEX_NEXT]]> = add vp<[[INDUCTION]]>, vp<[[VFxUF]]>
; CHECK-NEXT:     EMIT branch-on-count vp<[[INDEX_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ ir<%n>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val>.1 = phi [ ir<%n>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<for.body>
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %n, %entry ], [ %indvars.iv.next, %for.body ]
  %i.0.in8 = phi i32 [ %n, %entry ], [ %i.0, %for.body ]
  %i.0 = add nsw i32 %i.0.in8, -1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  %add9 = add i32 %1, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
  store i32 %add9, ptr %arrayidx3, align 4
  %cmp = icmp ugt i32 %indvars.iv, 1
  %indvars.iv.next = add nsw i32 %indvars.iv, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
