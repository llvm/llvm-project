; RUN: llc -march=amdgcn < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck %s

; Checks that we don't crash when code produces a build_vector with two undef operands.

; CHECK: {{^}}buildvector_undefs:
define amdgpu_kernel void @buildvector_undefs(<2 x i16> %in) {
entry:
  %i0 = call <16 x i16> @llvm.vector.insert.v16i16.v2i16(<16 x i16> poison, <2 x i16> %in, i64 0)
  %i1 = call <16 x i16> @llvm.vector.insert.v16i16.v2i16(<16 x i16> %i0, <2 x i16> zeroinitializer, i64 2)
  store <16 x i16> %i1, ptr addrspace(1) null, align 32
  ret void
}

declare <2 x i16> @llvm.vector.extract.v2i16.v16i16(<16 x i16>, i64 immarg)
declare <16 x i16> @llvm.vector.insert.v16i16.v2i16(<16 x i16>, <2 x i16>, i64 immarg)
