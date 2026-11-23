; RUN: split-file %s %t
; RUN: not opt -S -dxil-legalize -mtriple=dxil-pc-shadermodel6.0-compute %t/half.ll 2>&1 | FileCheck %t/half.ll
; RUN: not opt -S -dxil-legalize -mtriple=dxil-pc-shadermodel6.0-compute %t/fp128.ll 2>&1 | FileCheck %t/fp128.ll

; DXIL has 32-bit and 64-bit atomics only, so a float exchange of any other
; width has no integer exchange to lower to.

;--- half.ll

target triple = "dxil-pc-shadermodel6.0-compute"

@gs = external addrspace(3) global half

; CHECK: DXIL atomic exchange requires a 32-bit or 64-bit floating-point value
define half @gs_xchg_half(half %val) {
  %old = atomicrmw xchg ptr addrspace(3) @gs, half %val syncscope("workgroup") monotonic
  ret half %old
}

;--- fp128.ll

target triple = "dxil-pc-shadermodel6.0-compute"

@gs = external addrspace(3) global fp128

; CHECK: DXIL atomic exchange requires a 32-bit or 64-bit floating-point value
define fp128 @gs_xchg_fp128(fp128 %val) {
  %old = atomicrmw xchg ptr addrspace(3) @gs, fp128 %val syncscope("workgroup") monotonic
  ret fp128 %old
}
