; REQUIRES: asserts
; RUN: opt -mattr=+neon,+dotprod -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -enable-epilogue-vectorization -epilogue-vectorization-force-VF=2 -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; Tests for printing VPlans that are enabled under AArch64

define i32 @print_partial_reduction(ptr %a, ptr %b) {
; CHECK:      VPlan 'Initial VPlan for VF={8,16},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%.]]> = VF
; CHECK-NEXT: Live-in vp<[[VFxUF:%.]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<4>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<[[ACC:%.+]]> = phi vp<[[RDX_START]]>, vp<[[REDUCE:%.+]]> (VF scaled by 1/4)
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:   CLONE ir<%gep.a> = getelementptr ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[PTR_A:%.+]]> = vector-pointer ir<%gep.a>
; CHECK-NEXT:   WIDEN ir<%load.a> = load vp<[[PTR_A]]>
; CHECK-NEXT:   CLONE ir<%gep.b> = getelementptr ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[PTR_B:%.+]]> = vector-pointer ir<%gep.b>
; CHECK-NEXT:   WIDEN ir<%load.b> = load vp<[[PTR_B]]>
; CHECK-NEXT:   EXPRESSION vp<[[REDUCE]]> = ir<[[ACC]]> + partial.reduce.add (mul (ir<%load.b> zext to i32), (ir<%load.a> zext to i32))
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RESULT:%.+]]> = compute-reduction-result ir<[[ACC]]>, vp<[[REDUCE]]>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<1024>, vp<[[VEC_TC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<[[RED_RESULT]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<[[VEC_TC]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[RED_RESULT]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
; CHECK-NEXT:   IR   %accum = phi i32 [ 0, %entry ], [ %add, %for.body ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:   IR   %gep.a = getelementptr i8, ptr %a, i64 %iv
; CHECK-NEXT:   IR   %load.a = load i8, ptr %gep.a, align 1
; CHECK-NEXT:   IR   %ext.a = zext i8 %load.a to i32
; CHECK-NEXT:   IR   %gep.b = getelementptr i8, ptr %b, i64 %iv
; CHECK-NEXT:   IR   %load.b = load i8, ptr %gep.b, align 1
; CHECK-NEXT:   IR   %ext.b = zext i8 %load.b to i32
; CHECK-NEXT:   IR   %mul = mul i32 %ext.b, %ext.a
; CHECK-NEXT:   IR   %add = add i32 %mul, %accum
; CHECK-NEXT:   IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, 1024
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK: VPlan 'Final VPlan for VF={8,16},UF={1}' {
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.ph>:
; CHECK-NEXT:  EMIT vp<[[RDX_START:%.+]]> = reduction-start-vector ir<0>, ir<0>, ir<4>
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<[[EP_IV:%.+]]> = phi [ ir<0>, ir-bb<vector.ph> ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%accum> = phi vp<[[RDX_START]]>, ir<%add> (VF scaled by 1/4)
; CHECK-NEXT:   CLONE ir<%gep.a> = getelementptr ir<%a>, vp<[[EP_IV]]>
; CHECK-NEXT:   WIDEN ir<%load.a> = load ir<%gep.a>
; CHECK-NEXT:   CLONE ir<%gep.b> = getelementptr ir<%b>, vp<[[EP_IV]]>
; CHECK-NEXT:   WIDEN ir<%load.b> = load ir<%gep.b>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.b> = zext ir<%load.b> to i32
; CHECK-NEXT:   WIDEN-CAST ir<%ext.a> = zext ir<%load.a> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%ext.b>, ir<%ext.a>
; CHECK-NEXT:   PARTIAL-REDUCE ir<%add> = add ir<%accum>, ir<%mul>
; CHECK-NEXT:   EMIT vp<[[EP_IV_NEXT:%.+]]> = add nuw vp<[[EP_IV]]>, ir<16>
; CHECK-NEXT:   EMIT branch-on-count vp<[[EP_IV_NEXT]]>, ir<1024>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RESULT:%.+]]> = compute-reduction-result ir<%accum>, ir<%add>
; CHECK-NEXT:   EMIT branch-on-cond ir<true>
; CHECK-NEXT: Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<[[RED_RESULT]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<scalar.ph>:
; CHECK-NEXT:   EMIT-SCALAR vp<[[EP_RESUME:%.+]]> = phi [ ir<1024>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<[[EP_MERGE:%.+]]> = phi [ vp<[[RED_RESULT]]>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT-SCALAR vp<%6> = resume-for-epilogue vp<%vec.epilog.resume.val>
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %accum = phi i32 [ 0, %scalar.ph ], [ %add, %for.body ] (extra operand: vp<[[EP_MERGE]]> from ir-bb<scalar.ph>)
; CHECK-NEXT:   IR   %gep.a = getelementptr i8, ptr %a, i64 %iv
; CHECK-NEXT:   IR   %load.a = load i8, ptr %gep.a, align 1
; CHECK-NEXT:   IR   %ext.a = zext i8 %load.a to i32
; CHECK-NEXT:   IR   %gep.b = getelementptr i8, ptr %b, i64 %iv
; CHECK-NEXT:   IR   %load.b = load i8, ptr %gep.b, align 1
; CHECK-NEXT:   IR   %ext.b = zext i8 %load.b to i32
; CHECK-NEXT:   IR   %mul = mul i32 %ext.b, %ext.a
; CHECK-NEXT:   IR   %add = add i32 %mul, %accum
; CHECK-NEXT:   IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, 1024
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %accum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %gep.a = getelementptr i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a, align 1
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %ext.b = zext i8 %load.b to i32
  %mul = mul i32 %ext.b, %ext.a
  %add = add i32 %mul, %accum
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add
}

define i32 @print_partial_reduction_ext_mul(i64 %n, ptr %a, i8 %b) {
; CHECK:       VPlan 'Initial VPlan for VF={8},UF>=1' {
; CHECK-NEXT:  Live-in vp<%0> = VF * UF
; CHECK-NEXT:  Live-in vp<%1> = vector-trip-count
; CHECK-NEXT:  vp<%2> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%2> = EXPAND SCEV (1 + %n)
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<%3> = reduction-start-vector ir<0>, ir<0>, ir<4>
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<%4> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:      WIDEN-REDUCTION-PHI ir<%res2> = phi vp<%3>, vp<%5> (VF scaled by 1/4)
; CHECK-NEXT:      CLONE ir<%load> = load ir<%a>
; CHECK-NEXT:      EXPRESSION vp<%5> = ir<%res2> + partial.reduce.add (mul (ir<%b> zext to i32), (ir<%b> zext to i32))
; CHECK-NEXT:      WIDEN-CAST ir<%load.ext> = sext ir<%load> to i32
; CHECK-NEXT:      WIDEN-CAST ir<%load.ext.ext> = sext ir<%load.ext> to i64
; CHECK-NEXT:      EMIT vp<%index.next> = add nuw vp<%4>, vp<%0>
; CHECK-NEXT:      EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%7> = compute-reduction-result ir<%res2>, vp<%5>
; CHECK-NEXT:    EMIT vp<%vector.recur.extract> = extract-last-element ir<%load.ext.ext>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq vp<%2>, vp<%1>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %add.lcssa = phi i32 [ %add, %loop ] (extra operand: vp<%7> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%1>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%scalar.recur.init> = phi [ vp<%vector.recur.extract>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%7>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; CHECK-NEXT:    IR   %res1 = phi i64 [ 0, %entry ], [ %load.ext.ext, %loop ] (extra operand: vp<%scalar.recur.init> from scalar.ph)
; CHECK-NEXT:    IR   %res2 = phi i32 [ 0, %entry ], [ %add, %loop ] (extra operand: vp<%bc.merge.rdx> from scalar.ph)
; CHECK-NEXT:    IR   %load = load i16, ptr %a, align 2
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %conv = zext i8 %b to i16
; CHECK-NEXT:    IR   %mul = mul i16 %conv, %conv
; CHECK-NEXT:    IR   %mul.ext = zext i16 %mul to i32
; CHECK-NEXT:    IR   %add = add i32 %res2, %mul.ext
; CHECK-NEXT:    IR   %load.ext = sext i16 %load to i32
; CHECK-NEXT:    IR   %load.ext.ext = sext i32 %load.ext to i64
; CHECK-NEXT:    IR   %exitcond740.not = icmp eq i64 %iv, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }

; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<%0> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %0 = add i64 %n, 1
; CHECK-NEXT:    EMIT vp<%min.iters.check> = icmp ult ir<%0>, ir<8>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<%n.mod.vf> = urem ir<%0>, ir<8>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%0>, vp<%n.mod.vf>
; CHECK-NEXT:    EMIT vp<%3> = reduction-start-vector ir<0>, ir<0>, ir<4>
; CHECK-NEXT:    EMIT vp<%4> = broadcast ir<%b>
; CHECK-NEXT:  Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%res2> = phi vp<%3>, ir<%add> (VF scaled by 1/4)
; CHECK-NEXT:    CLONE ir<%load> = load ir<%a>
; CHECK-NEXT:    WIDEN-CAST vp<%5> = zext vp<%4> to i32
; CHECK-NEXT:    WIDEN ir<%mul> = mul vp<%5>, vp<%5>
; CHECK-NEXT:    PARTIAL-REDUCE ir<%add> = add ir<%res2>, ir<%mul>
; CHECK-NEXT:    WIDEN-CAST ir<%load.ext> = sext ir<%load> to i32
; CHECK-NEXT:    WIDEN-CAST ir<%load.ext.ext> = sext ir<%load.ext> to i64
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<8>
; CHECK-NEXT:    EMIT branch-on-count vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:  Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<%7> = compute-reduction-result ir<%res2>, ir<%add>
; CHECK-NEXT:    EMIT vp<%vector.recur.extract> = extract-last-element ir<%load.ext.ext>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%0>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %add.lcssa = phi i32 [ %add, %loop ] (extra operand: vp<%7> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%scalar.recur.init> = phi [ vp<%vector.recur.extract>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<%7>, middle.block ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %res1 = phi i64 [ 0, %scalar.ph ], [ %load.ext.ext, %loop ] (extra operand: vp<%scalar.recur.init> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %res2 = phi i32 [ 0, %scalar.ph ], [ %add, %loop ] (extra operand: vp<%bc.merge.rdx> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %load = load i16, ptr %a, align 2
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %conv = zext i8 %b to i16
; CHECK-NEXT:    IR   %mul = mul i16 %conv, %conv
; CHECK-NEXT:    IR   %mul.ext = zext i16 %mul to i32
; CHECK-NEXT:    IR   %add = add i32 %res2, %mul.ext
; CHECK-NEXT:    IR   %load.ext = sext i16 %load to i32
; CHECK-NEXT:    IR   %load.ext.ext = sext i32 %load.ext to i64
; CHECK-NEXT:    IR   %exitcond740.not = icmp eq i64 %iv, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %res1 = phi i64 [ 0, %entry ], [ %load.ext.ext, %loop ]
  %res2 = phi i32 [ 0, %entry ], [ %add, %loop ]
  %load = load i16, ptr %a, align 2
  %iv.next = add i64 %iv, 1
  %conv = zext i8 %b to i16
  %mul = mul i16 %conv, %conv
  %mul.ext = zext i16 %mul to i32
  %add = add i32 %res2, %mul.ext
  %load.ext = sext i16 %load to i32
  %load.ext.ext = sext i32 %load.ext to i64
  %exitcond740.not = icmp eq i64 %iv, %n
  br i1 %exitcond740.not, label %exit, label %loop

exit:
  ret i32 %add
}
