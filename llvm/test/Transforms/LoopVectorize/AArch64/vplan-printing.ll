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
; CHECK-NEXT: Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[CAN_IV_NEXT:%.+]]>
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<[[ACC:%.+]]> = phi ir<0>, ir<[[REDUCE:%.+]]> (VF scaled by 1/4)
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:   CLONE ir<%gep.a> = getelementptr ir<%a>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[PTR_A:%.+]]> = vector-pointer ir<%gep.a>
; CHECK-NEXT:   WIDEN ir<%load.a> = load vp<[[PTR_A]]>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.a> = zext ir<%load.a> to i32
; CHECK-NEXT:   CLONE ir<%gep.b> = getelementptr ir<%b>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[PTR_B:%.+]]> = vector-pointer ir<%gep.b>
; CHECK-NEXT:   WIDEN ir<%load.b> = load vp<[[PTR_B]]>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.b> = zext ir<%load.b> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%ext.b>, ir<%ext.a>
; CHECK-NEXT:   PARTIAL-REDUCE ir<[[REDUCE]]> = add ir<[[ACC]]>, ir<%mul>
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RESULT:%.+]]> = compute-reduction-result ir<[[ACC]]>, ir<[[REDUCE]]>
; CHECK-NEXT:   EMIT vp<[[EXTRACT:%.+]]> = extract-last-element vp<[[RED_RESULT]]>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<1024>, vp<[[VEC_TC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT:   EMIT vp<%bc.resume.val> = resume-phi vp<[[VEC_TC]]>, ir<0>
; CHECK-NEXT:   EMIT vp<%bc.merge.rdx> = resume-phi vp<[[RED_RESULT]]>, ir<0>
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
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<[[EXTRACT]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-NEXT: }
; CHECK: VPlan 'Final VPlan for VF={8,16},UF={1}' {
; CHECK-NEXT: Live-in ir<[[EP_VFxUF:.+]]> = VF * UF
; CHECK-NEXT: Live-in ir<[[EP_VEC_TC:.+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.ph>:
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[EP_IV:%.+]]> = phi [ ir<0>, ir-bb<vector.ph> ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:   WIDEN-REDUCTION-PHI ir<%accum> = phi ir<0>, ir<%add> (VF scaled by 1/4)
; CHECK-NEXT:   CLONE ir<%gep.a> = getelementptr ir<%a>, vp<[[EP_IV]]>
; CHECK-NEXT:   vp<[[PTR_A:%.+]]> = vector-pointer ir<%gep.a>
; CHECK-NEXT:   WIDEN ir<%load.a> = load vp<[[PTR_A]]>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.a> = zext ir<%load.a> to i32
; CHECK-NEXT:   CLONE ir<%gep.b> = getelementptr ir<%b>, vp<[[EP_IV]]>
; CHECK-NEXT:   vp<[[PTR_B:%.+]]> = vector-pointer ir<%gep.b>
; CHECK-NEXT:   WIDEN ir<%load.b> = load vp<[[PTR_B]]>
; CHECK-NEXT:   WIDEN-CAST ir<%ext.b> = zext ir<%load.b> to i32
; CHECK-NEXT:   WIDEN ir<%mul> = mul ir<%ext.b>, ir<%ext.a>
; CHECK-NEXT:   PARTIAL-REDUCE ir<%add> = add ir<%accum>, ir<%mul>
; CHECK-NEXT:   EMIT vp<[[EP_IV_NEXT:%.+]]> = add nuw vp<[[EP_IV]]>, ir<16>
; CHECK-NEXT:   EMIT branch-on-count vp<[[EP_IV_NEXT]]>, ir<1024>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RED_RESULT:%.+]]> = compute-reduction-result ir<%accum>, ir<%add>
; CHECK-NEXT:   EMIT vp<[[EXTRACT:%.+]]> = extract-last-element vp<[[RED_RESULT]]>
; CHECK-NEXT:   EMIT vp<[[CMP:%.+]]> = icmp eq ir<1024>, ir<1024>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<[[EXTRACT]]> from middle.block)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<scalar.ph>:
; CHECK-NEXT:   EMIT vp<[[EP_RESUME:%.+]]> = resume-phi ir<1024>, ir<0>
; CHECK-NEXT:   EMIT vp<[[EP_MERGE:%.+]]> = resume-phi vp<[[RED_RESULT]]>, ir<0>
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
