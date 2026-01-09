; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; TODO: This test currently fails with LLVM_ENABLE_EXPENSIVE_CHECKS enabled
; XFAIL: expensive_checks

; CHECK-SPIRV: OpName %[[#vec:]] "vec"
; CHECK-SPIRV: OpName %[[#index:]] "index"

; CHECK-SPIRV: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV: %[[#float2:]] = OpTypeVector %[[#float]] 2

; CHECK-SPIRV: %[[#res:]] = OpVectorExtractDynamic %[[#float]] %[[#vec]] %[[#index]]

define spir_kernel void @test(ptr addrspace(1) nocapture %out, <2 x float> %vec, i32 %index) {
entry:
  %res = extractelement <2 x float> %vec, i32 %index
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}
