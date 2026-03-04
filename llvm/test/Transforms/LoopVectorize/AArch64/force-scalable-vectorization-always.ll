; RUN: opt -passes=loop-vectorize -enable-epilogue-vectorization=false -scalable-vectorization=always -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64"

; Check that the cost of a fixed-VF is lower than that of comparable scalable VFs,
; but that a scalable VF is still chosen (due to the `-scalable-vectorization=always` flag)
define i32 @cost_prefers_fixed_width_vf_but_force_scalable_vf(ptr noalias %dst, ptr noalias %src, i64 %n) "target-cpu"="neoverse-n2" {
; CHECK: Checking a loop in 'cost_prefers_fixed_width_vf_but_force_scalable_vf'
; CHECK: Cost for VF 2: 11 (Estimated cost per lane: 5.5)
; CHECK: Cost for VF 4: 10 (Estimated cost per lane: 2.5)
; CHECK: Cost for VF 8: 10 (Estimated cost per lane: 1.2)
; CHECK: Cost for VF vscale x 1: Invalid (Estimated cost per lane: Invalid)
; CHECK: Cost for VF vscale x 2: 8 (Estimated cost per lane: 4.0)
; CHECK: Cost for VF vscale x 4: 10 (Estimated cost per lane: 2.5)
; CHECK: VPlan 'Final VPlan for VF={vscale x 4},UF={2}' {
entry:
  br label %loop

loop:
  %iv = phi i64 [ %n, %entry ], [ %iv.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %uniform.load = load i16, ptr %src, align 2
  %ext = sext i16 %uniform.load to i32
  %sum.next = add i32 %sum, %ext
  %dst.gep = getelementptr i16, ptr %dst, i64 %iv
  store i16 %uniform.load, ptr %dst.gep, align 2
  %iv.next = add i64 %iv, -1
  %cmp = icmp ugt i64 %iv, 0
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}

; Test that with '-scalable-vectorization=always', we still fall back to NEON
; if we can't vectorize with SVE (in this case, because SVE is unavailable)
define i32 @no_sve_fallback_to_neon(ptr %src, i64 %n) "target-features"="+neon" {
; CHECK: Checking a loop in 'no_sve_fallback_to_neon'
; CHECK: Cost for VF 2: 4 (Estimated cost per lane: 2.0)
; CHECK: Cost for VF 4: 4 (Estimated cost per lane: 1.0)
; CHECK-NOT: Cost for VF
; CHECK: VPlan 'Final VPlan for VF={2,4},UF={2}' {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %src.gep = getelementptr i32, ptr %src, i64 %iv
  %load = load i32, ptr %src.gep, align 4
  %sum.next = add i32 %sum, %load
  %iv.next = add i64 %iv, 1
  %cmp = icmp ult i64 %iv, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}
