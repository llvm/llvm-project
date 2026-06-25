; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck %s --implicit-check-not=scratch_ --implicit-check-not=buffer_
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -verify-machineinstrs < %s -o /dev/null

; The masked memory intrinsics (masked load/store, gather/scatter) are accepted
; on the "VGPR as memory" address space (13): AMDGPU never marks them legal, so
; ScalarizeMaskedMemIntrin rewrites them into element-wise loads/stores, which
; lower to reserved-VGPR register copies - no scratch or buffer memory.

@buf = internal addrspace(13) global [8 x i32] poison

; CHECK-LABEL: masked_ldst:
define amdgpu_kernel void @masked_ldst(ptr addrspace(1) %out, <4 x i32> %v, <4 x i1> %mask) {
  call void @llvm.masked.store.v4i32.p13(<4 x i32> %v, ptr addrspace(13) @buf, i32 4, <4 x i1> %mask)
  %l = call <4 x i32> @llvm.masked.load.v4i32.p13(ptr addrspace(13) @buf, i32 4, <4 x i1> %mask, <4 x i32> poison)
  store <4 x i32> %l, ptr addrspace(1) %out
  ret void
}

; A gather whose pointer vector is a splat of the global (no scalar operand
; naming it) must still be discovered by the layout pass and lower into the
; file - it does not fall back to an "empty file" diagnostic.
; CHECK-LABEL: gather_splat_base:
define amdgpu_kernel void @gather_splat_base(ptr addrspace(1) %out, <4 x i32> %idx, <4 x i1> %mask) {
  %vp = getelementptr i32, <4 x ptr addrspace(13)> splat (ptr addrspace(13) @buf), <4 x i32> %idx
  %g = call <4 x i32> @llvm.masked.gather.v4i32.v4p13(<4 x ptr addrspace(13)> %vp, i32 4, <4 x i1> %mask, <4 x i32> poison)
  store <4 x i32> %g, ptr addrspace(1) %out
  ret void
}

; Likewise when the pointer vector is a ConstantVector of constant-expression
; GEPs of the global (again no scalar operand names it).
; CHECK-LABEL: gather_constvec_base:
define amdgpu_kernel void @gather_constvec_base(ptr addrspace(1) %out, <4 x i1> %mask) {
  %g = call <4 x i32> @llvm.masked.gather.v4i32.v4p13(
    <4 x ptr addrspace(13)> <
      ptr addrspace(13) getelementptr (i32, ptr addrspace(13) @buf, i32 0),
      ptr addrspace(13) getelementptr (i32, ptr addrspace(13) @buf, i32 1),
      ptr addrspace(13) getelementptr (i32, ptr addrspace(13) @buf, i32 2),
      ptr addrspace(13) getelementptr (i32, ptr addrspace(13) @buf, i32 3)>,
    i32 4, <4 x i1> %mask, <4 x i32> poison)
  store <4 x i32> %g, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: gather_scatter:
define amdgpu_kernel void @gather_scatter(ptr addrspace(1) %out, <4 x i32> %v, <4 x i1> %mask) {
  %p0 = getelementptr i32, ptr addrspace(13) @buf, i32 0
  %p1 = getelementptr i32, ptr addrspace(13) @buf, i32 1
  %p2 = getelementptr i32, ptr addrspace(13) @buf, i32 2
  %p3 = getelementptr i32, ptr addrspace(13) @buf, i32 3
  %v0 = insertelement <4 x ptr addrspace(13)> poison, ptr addrspace(13) %p0, i32 0
  %v1 = insertelement <4 x ptr addrspace(13)> %v0, ptr addrspace(13) %p1, i32 1
  %v2 = insertelement <4 x ptr addrspace(13)> %v1, ptr addrspace(13) %p2, i32 2
  %vp = insertelement <4 x ptr addrspace(13)> %v2, ptr addrspace(13) %p3, i32 3
  call void @llvm.masked.scatter.v4i32.v4p13(<4 x i32> %v, <4 x ptr addrspace(13)> %vp, i32 4, <4 x i1> %mask)
  %g = call <4 x i32> @llvm.masked.gather.v4i32.v4p13(<4 x ptr addrspace(13)> %vp, i32 4, <4 x i1> %mask, <4 x i32> poison)
  store <4 x i32> %g, ptr addrspace(1) %out
  ret void
}
