; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@scalar1 = internal addrspace(3) global float 0.000000e+00, align 4
@scalar2 = internal addrspace(3) global float 0.000000e+00, align 4

; We shouldn't sink mul.rn.f32 to BB %merge because BB %merge post-dominates
; BB %entry. Over-sinking created more register pressure on this example. The
; backend would sink the fmuls to BB %merge, but not the loads for being
; conservative on sinking memory accesses. As a result, the loads and
; the two fmuls would be separated to two basic blocks, causing two
; cross-BB live ranges.
define float @post_dominate(float %x, i1 %cond) {
; CHECK-LABEL: post_dominate(
entry:
  %0 = load float, ptr addrspacecast (ptr addrspace(3) @scalar1 to ptr), align 4
  %1 = load float, ptr addrspacecast (ptr addrspace(3) @scalar2 to ptr), align 4
; CHECK: ld.shared.f32
; CHECK: ld.shared.f32
  %2 = fmul float %0, %0
  %3 = fmul float %1, %2
; CHECK-NOT: bra
; CHECK: mul.rn.f32
; CHECK: mul.rn.f32
  br i1 %cond, label %then, label %merge

then:
  %z = fadd float %x, %x
  br label %then2

then2:
  %z2 = fadd float %z, %z
  br label %merge

merge:
  %y = phi float [ 0.0, %entry ], [ %z2, %then2 ]
  %w = fadd float %y, %3
  ret float %w
}
