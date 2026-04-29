; RUN: opt -S -passes=loop-unroll -unroll-runtime < %s | FileCheck %s

; The trip count SCEV for "for (i = start; i < end; i++)" is
; (-1 * start) + end.  The SCEV expansion cost model should recognize that
; (-1 * X) is expanded as a negate (sub 0, X), not a real multiply.  On
; targets where i32 mul is expensive (e.g. AMDGPU quarter-rate), the old
; cost model would over-count this as a mul (cost 4) exceeding the default
; budget of 4, and reject runtime unrolling.  With the fix, it's costed as
; a sub (cost 1), well within budget.

; AMDGPU is used because its i32 mul has a cost of 4,
; making the over-counting observable against the default budget of 4.
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; CHECK-LABEL: @offset_start_loop
; The loop should be runtime-unrolled 8x (prologue + main unrolled body).
; CHECK: %xtraiter = and i32 %{{.*}}, 7
; CHECK: loop.prol:
; CHECK: loop:
; CHECK: %iv.next.7 = add nsw i32 %iv, 8
define void @offset_start_loop(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %start, i32 %end) {
entry:
  %cmp = icmp slt i32 %start, %end
  br i1 %cmp, label %loop, label %exit

loop:
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
  %idx = sext i32 %iv to i64
  %gep.in = getelementptr inbounds float, ptr addrspace(1) %in, i64 %idx
  %val = load float, ptr addrspace(1) %gep.in, align 4
  %gep.out = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx
  store float %val, ptr addrspace(1) %gep.out, align 4
  %iv.next = add nsw i32 %iv, 1
  %cond = icmp slt i32 %iv.next, %end
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
