; RUN: llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s

; Reduced from rocrand's threefry2x32_20 kernel.
; The AMDGPU LSR cost model should avoid creating a redundant VGPR induction
; variable when the loop already has a vector IV incremented by a uniform
; (SGPR) stride. Without the cost model fix, LSR introduces a second v_add
; in the loop body, wasting a VGPR and a VALU slot every iteration.

declare i32 @llvm.amdgcn.workitem.id.x() #0

; CHECK-LABEL: {{^}}lsr_vector_iv_cost:
; The loop must contain exactly one VALU add — the single vector IV update.
; A second v_add_u32 here would mean LSR created a redundant IV.
; CHECK:      {{^}}.LBB0_1:
; CHECK:      v_add_u32
; CHECK-NOT:  v_add_u32
; CHECK:      s_cbranch
define amdgpu_kernel void @lsr_vector_iv_cost(<2 x i32> %arg0, i32 %stride, ptr addrspace(1) %out) {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %iv.pn = phi i32 [ 0, %entry ], [ %or, %loop ]
  %iv.vec = phi i32 [ %tid, %entry ], [ %sum1, %loop ]
  %sum1 = add i32 %iv.vec, %stride
  %elt = extractelement <2 x i32> %arg0, i64 0
  %sum2 = add i32 %sum1, %elt
  %xor = xor i32 1, %sum2
  %sum3 = add i32 %sum2, %xor
  %sum4 = add i32 %sum3, %elt
  %or = or i32 %sum4, %stride
  %shr = lshr i32 %iv.pn, 1
  %cmp = icmp ult i32 %sum1, 1024
  br i1 %cmp, label %loop, label %exit

exit:
  store i32 %or, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
