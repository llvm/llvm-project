; REQUIRES: asserts
; RUN: opt -mattr=+neon,+dotprod -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -enable-epilogue-vectorization -epilogue-vectorization-force-VF=2 -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; Tests for printing VPlans that are enabled under AArch64
; CHECK: VPlan 'Final VPlan for VF={8,16},UF={1}' {
; CHECK-NEXT: Live-in ir<16> = VF * UF
; CHECK-NEXT: Live-in ir<1024> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.ph>:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     SCALAR-PHI vp<%0> = phi ir<0>, vp<%index.next>
; CHECK-NEXT:     WIDEN-REDUCTION-PHI ir<%accum> = phi ir<0>, vp<%4> (VF scaled by 1/4)
; CHECK-NEXT:     vp<%1> = SCALAR-STEPS vp<%0>, ir<1>
; CHECK-NEXT:     CLONE ir<%gep.a> = getelementptr ir<%a>, vp<%1>
; CHECK-NEXT:     vp<%2> = vector-pointer ir<%gep.a>
; CHECK-NEXT:     WIDEN ir<%load.a> = load vp<%2>
; CHECK-NEXT:     WIDEN-CAST ir<%ext.a> = zext ir<%load.a> to i32
; CHECK-NEXT:     CLONE ir<%gep.b> = getelementptr ir<%b>, vp<%1>
; CHECK-NEXT:     vp<%3> = vector-pointer ir<%gep.b>
; CHECK-NEXT:     WIDEN ir<%load.b> = load vp<%3>
; CHECK-NEXT:     WIDEN-CAST ir<%ext.b> = zext ir<%load.b> to i32
; CHECK-NEXT:     WIDEN ir<%mul> = mul ir<%ext.b>, ir<%ext.a>
; CHECK-NEXT:     PARTIAL-REDUCE vp<%4> = add ir<%mul>, ir<%accum>
; CHECK-NEXT:     EMIT vp<%index.next> = add nuw vp<%0>, ir<16>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, ir<1024>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): ir-bb<middle.block>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<middle.block>:
; CHECK-NEXT:   EMIT vp<%6> = compute-reduction-result ir<%accum>, vp<%4>
; CHECK-NEXT:   EMIT vp<%7> = extract-from-end vp<%6>, ir<1>
; CHECK-NEXT:   EMIT vp<%cmp.n> = icmp eq ir<1024>, ir<1024>
; CHECK-NEXT:   EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT: Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT:   IR   %add.lcssa = phi i32 [ %add, %for.body ] (extra operand: vp<%7> from ir-bb<middle.block>)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<scalar.ph>:
; CHECK-NEXT:   EMIT vp<%vec.epilog.resume.val> = resume-phi ir<1024>, ir<0>
; CHECK-NEXT:   EMIT vp<%bc.merge.rdx> = resume-phi vp<%6>, ir<0>
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %accum = phi i32 [ 0, %scalar.ph ], [ %add, %for.body ] (extra operand: vp<%bc.merge.rdx> from ir-bb<scalar.ph>)
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
define i32 @print_partial_reduction(ptr %a, ptr %b) {
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
