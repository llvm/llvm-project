; REQUIRES: asserts

; RUN: opt -loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Tests for printing VPlans.

define void @print_call_and_memory(i64 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
; CHECK: N0 [label =
; CHECK-NEXT: "for.body:\n" +
; CHECK-NEXT:       "WIDEN-INDUCTION %iv = phi %iv.next, 0\l" +
; CHECK-NEXT:       "CLONE ir<%arrayidx> = getelementptr ir<%y>, ir<%iv>\l" +
; CHECK-NEXT:       "WIDEN ir<%lv> = load ir<%arrayidx>\l" +
; CHECK-NEXT:       "WIDEN-CALL ir<%call> = call @llvm.sqrt.f32(ir<%lv>)\l" +
; CHECK-NEXT:       "CLONE ir<%arrayidx2> = getelementptr ir<%x>, ir<%iv>\l" +
; CHECK-NEXT:       "WIDEN store ir<%arrayidx2>, ir<%call>\l"
; CHECK-NEXT:   ]

entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %iv
  %lv = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.sqrt.f32(float %lv) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %iv
  store float %call, float* %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @print_widen_gep_and_select(i64 %n, float* noalias %y, float* noalias %x, float* %z) nounwind uwtable {
; CHECK: N0 [label =
; CHECK-NEXT: "for.body:\n" +
; CHECK-NEXT:      "WIDEN-INDUCTION %iv = phi %iv.next, 0\l" +
; CHECK-NEXT:      "WIDEN-GEP Inv[Var] ir<%arrayidx> = getelementptr ir<%y>, ir<%iv>\l" +
; CHECK-NEXT:      "WIDEN ir<%lv> = load ir<%arrayidx>\l" +
; CHECK-NEXT:      "WIDEN ir<%cmp> = icmp ir<%arrayidx>, ir<%z>\l" +
; CHECK-NEXT:      "WIDEN-SELECT ir<%sel> = select ir<%cmp>, ir<1.000000e+01>, ir<2.000000e+01>\l" +
; CHECK-NEXT:      "WIDEN ir<%add> = fadd ir<%lv>, ir<%sel>\l" +
; CHECK-NEXT:      "CLONE ir<%arrayidx2> = getelementptr ir<%x>, ir<%iv>\l" +
; CHECK-NEXT:      "WIDEN store ir<%arrayidx2>, ir<%add>\l"
; CHECK-NEXT:   ]

entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %iv
  %lv = load float, float* %arrayidx, align 4
  %cmp = icmp eq float* %arrayidx, %z
  %sel = select i1 %cmp, float 10.0, float 20.0
  %add = fadd float %lv, %sel
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %iv
  store float %add, float* %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define float @print_reduction(i64 %n, float* noalias %y) {
; CHECK: N0 [label =
; CHECK-NEXT: "for.body:\n" +
; CHECK-NEXT:       "WIDEN-INDUCTION %iv = phi %iv.next, 0\l" +
; CHECK-NEXT:       "WIDEN-PHI %red = phi %red.next, 0.000000e+00\l" +
; CHECK-NEXT:       "CLONE ir<%arrayidx> = getelementptr ir<%y>, ir<%iv>\l" +
; CHECK-NEXT:       "WIDEN ir<%lv> = load ir<%arrayidx>\l" +
; CHECK-NEXT:       "REDUCE ir<%red.next> = ir<%red> + reduce.fadd (ir<%lv>)\l"
; CHECK-NEXT:   ]

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %red = phi float [ %red.next, %for.body ], [ 0.0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %iv
  %lv = load float, float* %arrayidx, align 4
  %red.next = fadd fast float %lv, %red
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret float %red.next
}

define void @print_replicate_predicated_phi(i64 %n, i64* %x) {
; CHECK:       N0 [label =
; CHECK-NEXT:    "for.body:\n" +
; CHECK-NEXT:      "WIDEN-INDUCTION %i = phi 0, %i.next\l" +
; CHECK-NEXT:      "WIDEN ir<%cmp> = icmp ir<%i>, ir<5>\l"
; CHECK-NEXT:  ]
;
; CHECK:       N2 [label =
; CHECK-NEXT:    "pred.udiv.entry:\n" +
; CHECK-NEXT:      +
; CHECK-NEXT:      "BRANCH-ON-MASK ir<%cmp>\l"\l
; CHECK-NEXT:         "CondBit: ir<%cmp>"
; CHECK-NEXT:    ]
;
; CHECK:       N4 [label =
; CHECK-NEXT:    "pred.udiv.if:\n" +
; CHECK-NEXT:      "REPLICATE ir<%tmp4> = udiv ir<%n>, ir<%i> (S->V)\l"
; CHECK-NEXT:  ]
;
; CHECK:       N5 [label =
; CHECK-NEXT:    "pred.udiv.continue:\n" +
; CHECK-NEXT:      "PHI-PREDICATED-INSTRUCTION ir<%tmp4>\l"
; CHECK-NEXT:  ]
;
; CHECK:       N7 [label =
; CHECK-NEXT:    "for.inc:\n" +
; CHECK-NEXT:      "EMIT vp<%4> = not ir<%cmp>\l" +
; CHECK-NEXT:      "BLEND %d = ir<0>/vp<%4> ir<%tmp4>/ir<%cmp>\l" +
; CHECK-NEXT:      "CLONE ir<%idx> = getelementptr ir<%x>, ir<%i>\l" +
; CHECK-NEXT:      "WIDEN store ir<%idx>, ir<%d>\l"
; CHECK-NEXT:  ]
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %cmp = icmp ult i64 %i, 5
  br i1 %cmp, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %tmp4 = udiv i64 %n, %i
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %d = phi i64 [ 0, %for.body ], [ %tmp4, %if.then ]
  %idx = getelementptr i64, i64* %x, i64 %i
  store i64 %d, i64* %idx
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone
