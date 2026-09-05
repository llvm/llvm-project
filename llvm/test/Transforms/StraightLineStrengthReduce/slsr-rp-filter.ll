; REQUIRES: asserts
; RUN: opt -passes=slsr -stats -disable-output <%s 2>&1 | FileCheck %s

; CHECK-NOT: Number of blocks whose rewrites SLSR skipped due to register pressure

; The register-pressure filter needs a register budget to compare against, and
; the generic TargetTransformInfo has none: getRegisterBudget() returns
; std::nullopt unless a target implements it. There is no target triple here, so
; RPFilter::run() returns early, before it even computes liveness or pressure,
; and every rewrite stands.
;
; @many_basises_overlapping is the same body used by
; @peak_above_budget in AMDGPU/slsr-rp-filter.ll: 17 distinct basises whose
; rewrites take the block's peak pressure from 80 registers to 144. Under an
; AMDGPU triple that exceeds the budget and the rewrites are dropped.
;
; The check is on the statistic rather than on the IR because LLVM only prints a
; statistic that was incremented at least once, so a missing counter means the
; filter never skipped a block.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

declare void @bar(i128)

define void @many_basises_overlapping(i128 %s, i128 %b1, i128 %b2, i128 %b3, i128 %b4, i128 %b5, i128 %b6, i128 %b7, i128 %b8, i128 %b9, i128 %b10, i128 %b11, i128 %b12, i128 %b13, i128 %b14, i128 %b15, i128 %b16, i128 %b17) {
entry:
  %s2 = shl i128 %s, 1
  %t1 = add i128 %b1, %s
  call void @bar(i128 %t1)
  %t2 = add i128 %b2, %s
  call void @bar(i128 %t2)
  %t3 = add i128 %b3, %s
  call void @bar(i128 %t3)
  %t4 = add i128 %b4, %s
  call void @bar(i128 %t4)
  %t5 = add i128 %b5, %s
  call void @bar(i128 %t5)
  %t6 = add i128 %b6, %s
  call void @bar(i128 %t6)
  %t7 = add i128 %b7, %s
  call void @bar(i128 %t7)
  %t8 = add i128 %b8, %s
  call void @bar(i128 %t8)
  %t9 = add i128 %b9, %s
  call void @bar(i128 %t9)
  %t10 = add i128 %b10, %s
  call void @bar(i128 %t10)
  %t11 = add i128 %b11, %s
  call void @bar(i128 %t11)
  %t12 = add i128 %b12, %s
  call void @bar(i128 %t12)
  %t13 = add i128 %b13, %s
  call void @bar(i128 %t13)
  %t14 = add i128 %b14, %s
  call void @bar(i128 %t14)
  %t15 = add i128 %b15, %s
  call void @bar(i128 %t15)
  %t16 = add i128 %b16, %s
  call void @bar(i128 %t16)
  %t17 = add i128 %b17, %s
  call void @bar(i128 %t17)
  %u1 = add i128 %b1, %s2
  call void @bar(i128 %u1)
  %u2 = add i128 %b2, %s2
  call void @bar(i128 %u2)
  %u3 = add i128 %b3, %s2
  call void @bar(i128 %u3)
  %u4 = add i128 %b4, %s2
  call void @bar(i128 %u4)
  %u5 = add i128 %b5, %s2
  call void @bar(i128 %u5)
  %u6 = add i128 %b6, %s2
  call void @bar(i128 %u6)
  %u7 = add i128 %b7, %s2
  call void @bar(i128 %u7)
  %u8 = add i128 %b8, %s2
  call void @bar(i128 %u8)
  %u9 = add i128 %b9, %s2
  call void @bar(i128 %u9)
  %u10 = add i128 %b10, %s2
  call void @bar(i128 %u10)
  %u11 = add i128 %b11, %s2
  call void @bar(i128 %u11)
  %u12 = add i128 %b12, %s2
  call void @bar(i128 %u12)
  %u13 = add i128 %b13, %s2
  call void @bar(i128 %u13)
  %u14 = add i128 %b14, %s2
  call void @bar(i128 %u14)
  %u15 = add i128 %b15, %s2
  call void @bar(i128 %u15)
  %u16 = add i128 %b16, %s2
  call void @bar(i128 %u16)
  %u17 = add i128 %b17, %s2
  call void @bar(i128 %u17)
  call void @bar(i128 %b1)
  call void @bar(i128 %b2)
  call void @bar(i128 %b3)
  call void @bar(i128 %b4)
  call void @bar(i128 %b5)
  call void @bar(i128 %b6)
  call void @bar(i128 %b7)
  call void @bar(i128 %b8)
  call void @bar(i128 %b9)
  call void @bar(i128 %b10)
  call void @bar(i128 %b11)
  call void @bar(i128 %b12)
  call void @bar(i128 %b13)
  call void @bar(i128 %b14)
  call void @bar(i128 %b15)
  call void @bar(i128 %b16)
  call void @bar(i128 %b17)
  ret void
}
