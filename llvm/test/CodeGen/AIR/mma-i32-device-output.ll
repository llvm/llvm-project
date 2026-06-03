; Smoke test: an MMA kernel that reads i8 device inputs, runs a float
; simdgroup-matrix accumulate, and writes a genuinely i32 device output (the
; int8-dot pattern, where the f32 accumulator is fptosi'd to i32). The AIRWriter
; must keep the i32 output buffer typed as i32* rather than collapsing every
; device pointer in an MMA kernel to float*; otherwise the emitted GEP source
; type disagrees with the pointer pointee and the AIR reader rejects the module.
; Serializing to a valid metallib container (MTLB magic) exercises that path.
; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -c -N 4 %t.metallib | FileCheck %s

; CHECK: M T L B

@__tg_dot_ab = internal addrspace(3) global [64 x float] undef, align 4

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)

define void @mma_mixed(ptr addrspace(1) %qin, ptr addrspace(1) %out, i32 %idx) {
entry:
  %gq = getelementptr i8, ptr addrspace(1) %qin, i32 %idx
  %q = load i8, ptr addrspace(1) %gq, align 1
  %qf = sitofp i8 %q to float
  %sp = getelementptr float, ptr addrspace(3) @__tg_dot_ab, i32 %idx
  store float %qf, ptr addrspace(3) %sp, align 4
  %a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_ab, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %acc = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %a, <64 x float> %a)
  %e = extractelement <64 x float> %acc, i32 0
  %i = fptosi float %e to i32
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %idx
  store i32 %i, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @mma_mixed}
