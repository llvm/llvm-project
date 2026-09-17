; RUN: opt -S -dxil-legalize -mtriple=dxil-pc-shadermodel6.0-compute %s | FileCheck %s

; DXIL has no floating-point atomic op. A float exchange on groupshared memory
; only moves the bit pattern, so it becomes an i32 exchange with a bitcast on
; the value and on the result. Opaque pointers leave the pointer operand
; unchanged.

target triple = "dxil-pc-shadermodel6.0-compute"

@gs = external addrspace(3) global float

; CHECK-LABEL: define float @gs_xchg_float
define float @gs_xchg_float(float %val) {
  ; CHECK: [[CAST:%.*]] = bitcast float %val to i32
  ; CHECK: [[OLD:%.*]] = atomicrmw xchg ptr addrspace(3) @gs, i32 [[CAST]] syncscope("workgroup") monotonic
  ; CHECK: [[RES:%.*]] = bitcast i32 [[OLD]] to float
  %old = atomicrmw xchg ptr addrspace(3) @gs, float %val syncscope("workgroup") monotonic
  ; CHECK: ret float [[RES]]
  ret float %old
}

; An integer exchange must pass through with no bitcast.
; CHECK-LABEL: define i32 @gs_xchg_i32
define i32 @gs_xchg_i32(ptr addrspace(3) %p, i32 %val) {
  ; CHECK-NOT: bitcast
  ; CHECK: atomicrmw xchg ptr addrspace(3) %p, i32 %val syncscope("workgroup") monotonic
  %old = atomicrmw xchg ptr addrspace(3) %p, i32 %val syncscope("workgroup") monotonic
  ret i32 %old
}

; Only exchange is rewritten. DXIL does not support float atomic add anywhere,
; but fadd must not be silently turned into an integer add.
; CHECK-LABEL: define float @gs_fadd_float
define float @gs_fadd_float(ptr addrspace(3) %p, float %val) {
  ; CHECK-NOT: bitcast
  ; CHECK: atomicrmw fadd ptr addrspace(3) %p, float %val syncscope("workgroup") monotonic
  %old = atomicrmw fadd ptr addrspace(3) %p, float %val syncscope("workgroup") monotonic
  ret float %old
}
