; RUN: llc -mtriple=agx -mcpu=agx-g13g -stop-after=agx-strict-fp < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s -o /dev/null

; !agx.keep-denormals: a float operation that keeps denormals in a function
; that flushes them, without ieee-fp32. It gets the IEEE lowering (the hardware
; instruction, recomputed with integers when both operands are below 2^-101);
; the untagged addition stays a hardware addition.

; CHECK-LABEL: define agx_kernel void @tagged(
; CHECK:         %keep = fadd float %a, %b
; CHECK:         fcmp olt float {{.*}}, f0x0D000000
; CHECK:         br i1 {{.*}}, label %[[SLOW:[0-9]+]], label %[[JOIN:[0-9]+]]
; CHECK:       [[JOIN]]:
; CHECK-NEXT:    %[[SUM:[0-9]+]] = phi float [ %keep, %0 ], [ %{{[0-9]+}}, %[[SLOW]] ]
; CHECK-NEXT:    %flush = fadd float %[[SUM]], %b
; CHECK-NEXT:    store float %flush
define agx_kernel void @tagged(ptr addrspace(1) %out, float %a, float %b) #0 {
  %keep = fadd float %a, %b, !agx.keep-denormals !0
  %flush = fadd float %keep, %b
  store float %flush, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign" }
!0 = !{}
