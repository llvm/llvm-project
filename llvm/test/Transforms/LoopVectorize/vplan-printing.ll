; REQUIRES: asserts

; RUN: opt -loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -enable-interleaved-mem-accesses=true -enable-masked-interleaved-mem-accesses -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Tests for printing VPlans.

define void @print_call_and_memory(i64 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
; CHECK-LABEL: Checking a loop in "print_call_and_memory"
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: for.body:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi %iv.next, 0
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr ir<%y>, ir<%iv>
; CHECK-NEXT:   WIDEN ir<%lv> = load ir<%arrayidx>
; CHECK-NEXT:   WIDEN-CALL ir<%call> = call @llvm.sqrt.f32(ir<%lv>)
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr ir<%x>, ir<%iv>
; CHECK-NEXT:   WIDEN store ir<%arrayidx2>, ir<%call>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
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
; CHECK-LABEL: Checking a loop in "print_widen_gep_and_select"
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: for.body:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi %iv.next, 0
; CHECK-NEXT:   WIDEN-GEP Inv[Var] ir<%arrayidx> = getelementptr ir<%y>, ir<%iv>
; CHECK-NEXT:   WIDEN ir<%lv> = load ir<%arrayidx>
; CHECK-NEXT:   WIDEN ir<%cmp> = icmp ir<%arrayidx>, ir<%z>
; CHECK-NEXT:   WIDEN-SELECT ir<%sel> = select ir<%cmp>, ir<1.000000e+01>, ir<2.000000e+01>
; CHECK-NEXT:   WIDEN ir<%add> = fadd ir<%lv>, ir<%sel>
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr ir<%x>, ir<%iv>
; CHECK-NEXT:   WIDEN store ir<%arrayidx2>, ir<%add>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
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
; CHECK-LABEL: Checking a loop in "print_reduction"
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: for.body:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi %iv.next, 0
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%red> = phi ir<0.000000e+00>, ir<%red.next>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr ir<%y>, ir<%iv>
; CHECK-NEXT:   WIDEN ir<%lv> = load ir<%arrayidx>
; CHECK-NEXT:   REDUCE ir<%red.next> = ir<%red> + fast reduce.fadd (ir<%lv>)
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
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
; CHECK-LABEL: Checking a loop in "print_replicate_predicated_phi"
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: for.body:
; CHECK-NEXT:   WIDEN-INDUCTION %i = phi 0, %i.next
; CHECK-NEXT:   WIDEN ir<%cmp> = icmp ir<%i>, ir<5>
; CHECK-NEXT: Successor(s): if.then
; CHECK-EMPTY:
; CHECK-NEXT: if.then:
; CHECK-NEXT: Successor(s): pred.udiv
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.udiv: {
; CHECK-NEXT:   pred.udiv.entry:
; CHECK-NEXT:     BRANCH-ON-MASK ir<%cmp>
; CHECK-NEXT:   Successor(s): pred.udiv.if, pred.udiv.continue
; CHECK-NEXT:   CondBit: ir<%cmp>
; CHECK-EMPTY:
; CHECK-NEXT:   pred.udiv.if:
; CHECK-NEXT:     REPLICATE ir<%tmp4> = udiv ir<%n>, ir<%i> (S->V)
; CHECK-NEXT:   Successor(s): pred.udiv.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.udiv.continue:
; CHECK-NEXT:     PHI-PREDICATED-INSTRUCTION vp<[[PRED:%.+]]> = ir<%tmp4>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): if.then.0
; CHECK-EMPTY:
; CHECK-NEXT: if.then.0:
; CHECK-NEXT: Successor(s): for.inc
; CHECK-EMPTY:
; CHECK-NEXT: for.inc:
; CHECK-NEXT:   EMIT vp<[[NOT:%.+]]> = not ir<%cmp>
; CHECK-NEXT:   BLEND %d = ir<0>/vp<[[NOT]]> vp<[[PRED]]>/ir<%cmp>
; CHECK-NEXT:   CLONE ir<%idx> = getelementptr ir<%x>, ir<%i>
; CHECK-NEXT:   WIDEN store ir<%idx>, ir<%d>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
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

@AB = common global [1024 x i32] zeroinitializer, align 4
@CD = common global [1024 x i32] zeroinitializer, align 4

define void @print_interleave_groups(i32 %C, i32 %D) {
; CHECK-LABEL: Checking a loop in "print_interleave_groups"
; CHECK:       VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  for.body:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   CLONE ir<%gep.AB.0> = getelementptr ir<@AB>, ir<0>, ir<%iv>
; CHECK-NEXT:   INTERLEAVE-GROUP with factor 4 at %AB.0, ir<%gep.AB.0>
; CHECK-NEXT:     ir<%AB.0> = load from index 0
; CHECK-NEXT:     ir<%AB.1> = load from index 1
; CHECK-NEXT:     ir<%AB.3> = load from index 3
; CHECK-NEXT:   CLONE ir<%iv.plus.1> = add ir<%iv>, ir<1>
; CHECK-NEXT:   CLONE ir<%gep.AB.1> = getelementptr ir<@AB>, ir<0>, ir<%iv.plus.1>
; CHECK-NEXT:   CLONE ir<%iv.plus.2> = add ir<%iv>, ir<2>
; CHECK-NEXT:   CLONE ir<%iv.plus.3> = add ir<%iv>, ir<3>
; CHECK-NEXT:   CLONE ir<%gep.AB.3> = getelementptr ir<@AB>, ir<0>, ir<%iv.plus.3>
; CHECK-NEXT:   WIDEN ir<%add> = add ir<%AB.0>, ir<%AB.1>
; CHECK-NEXT:   CLONE ir<%gep.CD.0> = getelementptr ir<@CD>, ir<0>, ir<%iv>
; CHECK-NEXT:   CLONE ir<%gep.CD.1> = getelementptr ir<@CD>, ir<0>, ir<%iv.plus.1>
; CHECK-NEXT:   CLONE ir<%gep.CD.2> = getelementptr ir<@CD>, ir<0>, ir<%iv.plus.2>
; CHECK-NEXT:   CLONE ir<%gep.CD.3> = getelementptr ir<@CD>, ir<0>, ir<%iv.plus.3>
; CHECK-NEXT:   INTERLEAVE-GROUP with factor 4 at <badref>, ir<%gep.CD.3>
; CHECK-NEXT:     store ir<%add> to index 0
; CHECK-NEXT:     store ir<1> to index 1
; CHECK-NEXT:     store ir<2> to index 2
; CHECK-NEXT:     store ir<%AB.3> to index 3
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.AB.0= getelementptr inbounds [1024 x i32], [1024 x i32]* @AB, i64 0, i64 %iv
  %AB.0 = load i32, i32* %gep.AB.0, align 4
  %iv.plus.1 = add i64 %iv, 1
  %gep.AB.1 = getelementptr inbounds [1024 x i32], [1024 x i32]* @AB, i64 0, i64 %iv.plus.1
  %AB.1 = load i32, i32* %gep.AB.1, align 4
  %iv.plus.2 = add i64 %iv, 2
  %iv.plus.3 = add i64 %iv, 3
  %gep.AB.3 = getelementptr inbounds [1024 x i32], [1024 x i32]* @AB, i64 0, i64 %iv.plus.3
  %AB.3 = load i32, i32* %gep.AB.3, align 4
  %add = add nsw i32 %AB.0, %AB.1
  %gep.CD.0 = getelementptr inbounds [1024 x i32], [1024 x i32]* @CD, i64 0, i64 %iv
  store i32 %add, i32* %gep.CD.0, align 4
  %gep.CD.1 = getelementptr inbounds [1024 x i32], [1024 x i32]* @CD, i64 0, i64 %iv.plus.1
  store i32 1, i32* %gep.CD.1, align 4
  %gep.CD.2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @CD, i64 0, i64 %iv.plus.2
  store i32 2, i32* %gep.CD.2, align 4
  %gep.CD.3 = getelementptr inbounds [1024 x i32], [1024 x i32]* @CD, i64 0, i64 %iv.plus.3
  store i32 %AB.3, i32* %gep.CD.3, align 4
  %iv.next = add nuw nsw i64 %iv, 4
  %cmp = icmp slt i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define float @print_fmuladd_strict(float* %a, float* %b, i64 %n) {
; CHECK-LABEL: Checking a loop in "print_fmuladd_strict"
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: for.body:
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%sum.07> = phi ir<0.000000e+00>, ir<%muladd>
; CHECK-NEXT:   CLONE ir<%arrayidx> = getelementptr ir<%a>, ir<%iv>
; CHECK-NEXT:   WIDEN ir<%l.a> = load ir<%arrayidx>
; CHECK-NEXT:   CLONE ir<%arrayidx2> = getelementptr ir<%b>, ir<%iv>
; CHECK-NEXT:   WIDEN ir<%l.b> = load ir<%arrayidx2>
; CHECK-NEXT:   EMIT vp<[[FMUL:%.]]> = fmul nnan ninf nsz ir<%l.a> ir<%l.b>
; CHECK-NEXT:   REDUCE ir<[[MULADD:%.+]]> = ir<%sum.07> + nnan ninf nsz reduce.fadd (vp<[[FMUL]]>)
; CHECK-NEXT:   No successors
; CHECK-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %l.a = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %l.b = load float, float* %arrayidx2, align 4
  %muladd = tail call nnan ninf nsz float @llvm.fmuladd.f32(float %l.a, float %l.b, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %muladd
}

define void @debug_loc_vpinstruction(i32* nocapture %asd, i32* nocapture %bsd) !dbg !5 {
; CHECK-LABEL: Checking a loop in "debug_loc_vpinstruction"
; CHECK:    VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  loop:
; CHECK-NEXT:    WIDEN-INDUCTION %iv = phi 0, %iv.next
; CHECK-NEXT:    CLONE ir<%isd> = getelementptr ir<%asd>, ir<%iv>
; CHECK-NEXT:    WIDEN ir<%lsd> = load ir<%isd>
; CHECK-NEXT:    WIDEN ir<%psd> = add ir<%lsd>, ir<23>
; CHECK-NEXT:    WIDEN ir<%cmp1> = icmp ir<%lsd>, ir<100>
; CHECK-NEXT:  Successor(s): check
; CHECK-EMPTY:
; CHECK-NEXT:  check:
; CHECK-NEXT:    WIDEN ir<%cmp2> = icmp ir<%lsd>, ir<200>
; CHECK-NEXT:  Successor(s): if.then
; CHECK-EMPTY:
; CHECK-NEXT:  if.then:
; CHECK-NEXT:    EMIT vp<[[NOT1:%.+]]> = not ir<%cmp1>, !dbg /tmp/s.c:5:3
; CHECK-NEXT:    EMIT vp<[[SEL1:%.+]]> = select vp<[[NOT1]]> ir<%cmp2> ir<false>, !dbg /tmp/s.c:5:21
; CHECK-NEXT:    EMIT vp<[[OR1:%.+]]> = or vp<[[SEL1]]> ir<%cmp1>
; CHECK-NEXT:  Successor(s): pred.sdiv
; CHECK-EMPTY:
; CHECK-NEXT:  <xVFxUF> pred.sdiv: {
; CHECK-NEXT:    pred.sdiv.entry:
; CHECK-NEXT:      BRANCH-ON-MASK vp<[[OR1]]>
; CHECK-NEXT:    Successor(s): pred.sdiv.if, pred.sdiv.continue
; CHECK-NEXT:    CondBit: vp<[[OR1]]> (if.then)
; CHECK-EMPTY:
; CHECK-NEXT:    pred.sdiv.if:
; CHECK-NEXT:      REPLICATE ir<%sd1> = sdiv ir<%psd>, ir<%lsd> (S->V)
; CHECK-NEXT:    Successor(s): pred.sdiv.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.sdiv.continue:
; CHECK-NEXT:      PHI-PREDICATED-INSTRUCTION vp<[[PHI:%.+]]> = ir<%sd1>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): if.then.0
; CHECK-EMPTY:
; CHECK-NEXT:  if.then.0:
; CHECK-NEXT:  Successor(s): if.end
; CHECK-EMPTY:
; CHECK-NEXT:  if.end:
; CHECK-NEXT:    EMIT vp<[[NOT2:%.+]]> = not ir<%cmp2>
; CHECK-NEXT:    EMIT vp<[[SEL2:%.+]]> = select vp<[[NOT1]]> vp<[[NOT2]]> ir<false>
; CHECK-NEXT:    BLEND %ysd.0 = vp<[[PHI]]>/vp<[[OR1]]> ir<%psd>/vp<[[SEL2]]>
; CHECK-NEXT:    WIDEN store ir<%isd>, ir<%ysd.0>
; CHECK-NEXT:  No successors
; CHECK-NEXT:}
; CHECK-NEXT:No successors
; CHECK-NEXT:}
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %if.end ]
  %isd = getelementptr inbounds i32, i32* %asd, i64 %iv
  %lsd = load i32, i32* %isd, align 4
  %psd = add nsw i32 %lsd, 23
  %cmp1 = icmp slt i32 %lsd, 100
  br i1 %cmp1, label %if.then, label %check, !dbg !7

check:
  %cmp2 = icmp sge i32 %lsd, 200
  br i1 %cmp2, label %if.then, label %if.end, !dbg !8

if.then:
  %sd1 = sdiv i32 %psd, %lsd
  br label %if.end

if.end:
  %ysd.0 = phi i32 [ %sd1, %if.then ], [ %psd, %check ]
  store i32 %ysd.0, i32* %isd, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 128
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone
declare float @llvm.fmuladd.f32(float, float, float)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !6, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 5, column: 3, scope: !5)
!8 = !DILocation(line: 5, column: 21, scope: !5)
