; LLVM IR allows alignments up to 2^32, but SPIR-V encodes memory access
; alignments as 32-bit literals. Check that they are clamped to 2^31 instead of
; truncated to 0.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-LABEL: Begin function scalar
; CHECK: OpLoad %[[#]] %[[#]] Aligned 2147483648
; CHECK: OpStore %[[#]] %[[#]] Aligned 2147483648
define spir_func void @scalar() {
  %v = load float, ptr null, align 4294967296
  store float %v, ptr null, align 4294967296
  ret void
}

; CHECK-LABEL: Begin function aggregate
; CHECK: OpLoad %[[#]] %[[#]] Aligned 2147483648
; CHECK: OpStore %[[#]] %[[#]] Aligned 2147483648
define spir_func void @aggregate() {
  %v = load { i32, i32 }, ptr null, align 4294967296
  store { i32, i32 } %v, ptr null, align 4294967296
  ret void
}

; CHECK-LABEL: Begin function memcpy
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]] Aligned 2147483648
define spir_func void @memcpy(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4294967296 %dst, ptr addrspace(1) align 4294967296 %src, i64 %n, i1 false)
  ret void
}
