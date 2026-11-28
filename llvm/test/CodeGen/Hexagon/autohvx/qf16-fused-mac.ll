; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Fused multiply-accumulate on qf16: the vmpy product stays in qf16 and is
; accumulated with vadd_qf16_mix, so a MAC needs one conversion back to hf
; instead of two.

; The shape that matters: a loop-carried accumulator, which is what a vectorized
; f16 matmul inner loop looks like. Without the fma pattern the product is
; converted back to hf before the add (vadd(hf,hf)), costing an extra
; instruction per MAC.
define <64 x half> @mac_loop_carried(<64 x half> %init, ptr %pa, ptr %pb, i32 %n) #0 {
; CHECK-LABEL: mac_loop_carried:
; CHECK:         [[M:v[0-9]+]].qf16 = vmpy({{v[0-9]+}}.hf,{{v[0-9]+}}.hf)
; CHECK:         [[A:v[0-9]+]].qf16 = vadd([[M]].qf16,{{v[0-9]+}}.hf)
; CHECK-NOT:     .hf = [[M]].qf16
entry:
  br label %loop

loop:
  %acc = phi <64 x half> [ %init, %entry ], [ %new, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %a = load <64 x half>, ptr %pa
  %b = load <64 x half>, ptr %pb
  %mul = fmul fast <64 x half> %a, %b
  %new = fadd fast <64 x half> %acc, %mul
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret <64 x half> %new
}

; An explicit fma is matched directly, with no fast-math flags needed.
define <64 x half> @fma_v64f16(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2) #0 {
; CHECK-LABEL: fma_v64f16:
; CHECK:         [[M:v[0-9]+]].qf16 = vmpy(v0.hf,v1.hf)
; CHECK:         [[A:v[0-9]+]].qf16 = vadd([[M]].qf16,v2.hf)
; CHECK:         v0.hf = [[A]].qf16
  %v0 = call <64 x half> @llvm.fma.v64f16(<64 x half> %a0, <64 x half> %a1, <64 x half> %a2)
  ret <64 x half> %v0
}

declare <64 x half> @llvm.fma.v64f16(<64 x half>, <64 x half>, <64 x half>)

attributes #0 = { nounwind "target-cpu"="hexagonv68" "target-features"="+hvxv68,+hvx-length128b,+hvx-qfloat" }
