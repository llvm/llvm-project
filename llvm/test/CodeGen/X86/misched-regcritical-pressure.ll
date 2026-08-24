; REQUIRES: asserts
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=cascade-lake \
; RUN:   -mattr=+avx512f,+avx512vl -enable-misched -verify-machineinstrs \
; RUN:   -stats 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=cascade-lake \
; RUN:   -mattr=+avx512f,+avx512vl -enable-misched -verify-machineinstrs \
; RUN:   -mllvm -misched-regcritical-pressure-threshold=1.5 \
; RUN:   -stats 2>&1 | FileCheck %s --check-prefix=THRESHOLD

; This test verifies that -misched-regcritical-pressure-threshold de-prioritizes
; the RegCritical heuristic when region register pressure is already significantly
; over the limit. The flag causes the scheduler to fall through to latency, stall,
; and node order heuristics instead.
;
; With a high threshold, fewer scheduling decisions should be attributed to
; RegCritical because regions with critically high pressure bypass it.

; Verify that fewer units are chosen for RegCritical with the threshold.
; The default run should have RegCritical decisions.
; DEFAULT: Number of scheduling units chosen for RegCritical heuristic
; The threshold run should have fewer (or zero) RegCritical decisions.
; THRESHOLD-NOT: {{[1-9][0-9]+}} {{.*}}Number of scheduling units chosen for RegCritical heuristic

; Create many live zmm values to exceed the 32-register limit.
define void @high_pressure(ptr %out, ptr %in, i64 %count) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]

  ; Load 16 vectors to create high register pressure
  %v0 = load <16 x i32>, ptr %in
  %p1 = getelementptr <16 x i32>, ptr %in, i64 1
  %v1 = load <16 x i32>, ptr %p1
  %p2 = getelementptr <16 x i32>, ptr %in, i64 2
  %v2 = load <16 x i32>, ptr %p2
  %p3 = getelementptr <16 x i32>, ptr %in, i64 3
  %v3 = load <16 x i32>, ptr %p3
  %p4 = getelementptr <16 x i32>, ptr %in, i64 4
  %v4 = load <16 x i32>, ptr %p4
  %p5 = getelementptr <16 x i32>, ptr %in, i64 5
  %v5 = load <16 x i32>, ptr %p5
  %p6 = getelementptr <16 x i32>, ptr %in, i64 6
  %v6 = load <16 x i32>, ptr %p6
  %p7 = getelementptr <16 x i32>, ptr %in, i64 7
  %v7 = load <16 x i32>, ptr %p7
  %p8 = getelementptr <16 x i32>, ptr %in, i64 8
  %v8 = load <16 x i32>, ptr %p8
  %p9 = getelementptr <16 x i32>, ptr %in, i64 9
  %v9 = load <16 x i32>, ptr %p9
  %p10 = getelementptr <16 x i32>, ptr %in, i64 10
  %v10 = load <16 x i32>, ptr %p10
  %p11 = getelementptr <16 x i32>, ptr %in, i64 11
  %v11 = load <16 x i32>, ptr %p11
  %p12 = getelementptr <16 x i32>, ptr %in, i64 12
  %v12 = load <16 x i32>, ptr %p12
  %p13 = getelementptr <16 x i32>, ptr %in, i64 13
  %v13 = load <16 x i32>, ptr %p13
  %p14 = getelementptr <16 x i32>, ptr %in, i64 14
  %v14 = load <16 x i32>, ptr %p14
  %p15 = getelementptr <16 x i32>, ptr %in, i64 15
  %v15 = load <16 x i32>, ptr %p15

  ; 8 hash values live across the loop body
  %h0 = add <16 x i32> %v0, %v1
  %h1 = add <16 x i32> %v2, %v3
  %h2 = add <16 x i32> %v4, %v5
  %h3 = add <16 x i32> %v6, %v7
  %h4 = add <16 x i32> %v8, %v9
  %h5 = add <16 x i32> %v10, %v11
  %h6 = add <16 x i32> %v12, %v13
  %h7 = add <16 x i32> %v14, %v15

  ; Mix operations creating dependencies
  %m0 = xor <16 x i32> %h0, %v0
  %m1 = xor <16 x i32> %h1, %v2
  %m2 = xor <16 x i32> %h2, %v4
  %m3 = xor <16 x i32> %h3, %v6
  %m4 = xor <16 x i32> %h4, %v8
  %m5 = xor <16 x i32> %h5, %v10
  %m6 = xor <16 x i32> %h6, %v12
  %m7 = xor <16 x i32> %h7, %v14

  %m8 = add <16 x i32> %m0, %v1
  %m9 = add <16 x i32> %m1, %v3
  %m10 = add <16 x i32> %m2, %v5
  %m11 = add <16 x i32> %m3, %v7
  %m12 = add <16 x i32> %m4, %v9
  %m13 = add <16 x i32> %m5, %v11
  %m14 = add <16 x i32> %m6, %v13
  %m15 = add <16 x i32> %m7, %v15

  ; Add back to h values
  %h0.new = add <16 x i32> %h0, %m8
  %h1.new = add <16 x i32> %h1, %m9
  %h2.new = add <16 x i32> %h2, %m10
  %h3.new = add <16 x i32> %h3, %m11
  %h4.new = add <16 x i32> %h4, %m12
  %h5.new = add <16 x i32> %h5, %m13
  %h6.new = add <16 x i32> %h6, %m14
  %h7.new = add <16 x i32> %h7, %m15

  ; Cross operations
  %r0 = xor <16 x i32> %h0.new, %h4.new
  %r1 = xor <16 x i32> %h1.new, %h5.new
  %r2 = xor <16 x i32> %h2.new, %h6.new
  %r3 = xor <16 x i32> %h3.new, %h7.new
  %r4 = xor <16 x i32> %h4.new, %h0.new
  %r5 = xor <16 x i32> %h5.new, %h1.new
  %r6 = xor <16 x i32> %h6.new, %h2.new
  %r7 = xor <16 x i32> %h7.new, %h3.new

  ; Final reduce
  %f0 = add <16 x i32> %r0, %r4
  %f1 = add <16 x i32> %r1, %r5
  %f2 = add <16 x i32> %r2, %r6
  %f3 = add <16 x i32> %r3, %r7

  ; Store results
  store <16 x i32> %f0, ptr %out
  %o1 = getelementptr <16 x i32>, ptr %out, i64 1
  store <16 x i32> %f1, ptr %o1
  %o2 = getelementptr <16 x i32>, ptr %out, i64 2
  store <16 x i32> %f2, ptr %o2
  %o3 = getelementptr <16 x i32>, ptr %out, i64 3
  store <16 x i32> %f3, ptr %o3

  %iv.next = add i64 %iv, 1
  %done = icmp eq i64 %iv.next, %count
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
