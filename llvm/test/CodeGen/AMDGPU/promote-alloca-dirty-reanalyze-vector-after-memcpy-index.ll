; RUN: opt -S -verify-each -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-promote-alloca < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+promote-alloca -filetype=null < %s
;
; Earlier vector promotion can collapse a later alloca's memcpy index from a
; dynamic load to a constant. Re-analyzing the dirty later alloca lets it stay
; on the vector path instead of falling back to LDS.

declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture,
                                    ptr addrspace(5) nocapture, i64,
                                    i1 immarg)

define amdgpu_kernel void @dirty_reanalyze_vector_after_memcpy_index(
    ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @dirty_reanalyze_vector_after_memcpy_index(
; CHECK-NOT: alloca [4 x i32]
; CHECK-NOT: @dirty_reanalyze_vector_after_memcpy_index.c
; CHECK-DAG: %b = freeze <4 x i32> poison
; CHECK-DAG: %c = freeze <4 x i32> poison
; CHECK: ret void
entry:
  %b = alloca [4 x i32], align 4, addrspace(5)
  %c = alloca [4 x i32], align 4, addrspace(5)

  ; Make %b score highly enough that it is promoted first, and let its final
  ; value collapse to the constant 1 after vector promotion.
  %b0 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 0
  store i32 1, ptr addrspace(5) %b0, align 4
  %bval0 = load i32, ptr addrspace(5) %b0, align 4
  %b1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 1
  store i32 %bval0, ptr addrspace(5) %b1, align 4
  %bval1 = load i32, ptr addrspace(5) %b1, align 4
  %b2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 2
  store i32 %bval1, ptr addrspace(5) %b2, align 4
  %bval = load i32, ptr addrspace(5) %b2, align 4

  ; %c is only LDS-promotable during the initial analysis because its memcpy
  ; source index is dynamic. After %b is promoted, %bval becomes a constant and
  ; re-analyzing %c should keep it on the vector path too.
  %c0 = getelementptr inbounds [4 x i32], ptr addrspace(5) %c, i32 0, i32 0
  store i32 11, ptr addrspace(5) %c0, align 4
  %c1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %c, i32 0, i32 1
  store i32 22, ptr addrspace(5) %c1, align 4
  %cdst = getelementptr inbounds [4 x i32], ptr addrspace(5) %c, i32 0, i32 2
  %csrc = getelementptr inbounds [4 x i32], ptr addrspace(5) %c, i32 0, i32 %bval
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 4 %cdst,
                                   ptr addrspace(5) align 4 %csrc, i64 8,
                                   i1 false)
  %cout = load i32, ptr addrspace(5) %cdst, align 4
  store i32 %cout, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "amdgpu-promote-alloca-to-vector-max-regs"="1024" }
