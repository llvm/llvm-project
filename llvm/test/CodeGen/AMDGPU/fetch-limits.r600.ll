; RUN: llc < %s -march=r600 -mcpu=r600 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=rs880 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=rv670 | FileCheck %s

; R600 supports 8 fetches in a clause
; CHECK: {{^}}fetch_limits_r600:
; CHECK: Fetch clause
; CHECK: Fetch clause

define amdgpu_ps void @fetch_limits_r600() {
entry:
  %0 = load <4 x float>, <4 x float> addrspace(8)* null
  %1 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %2 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %3 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %4 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %5 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 5)
  %6 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 6)
  %7 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 7)
  %8 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 8)
  %res0 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %0, i32 0, i32 0, i32 1)
  %res1 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %1, i32 0, i32 0, i32 1)
  %res2 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %2, i32 0, i32 0, i32 1)
  %res3 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %3, i32 0, i32 0, i32 1)
  %res4 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %4, i32 0, i32 0, i32 1)
  %res5 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %5, i32 0, i32 0, i32 1)
  %res6 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %6, i32 0, i32 0, i32 1)
  %res7 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %7, i32 0, i32 0, i32 1)
  %res8 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 0, i32 0, i32 1)
  %a = fadd <4 x float> %res0, %res1
  %b = fadd <4 x float> %res2, %res3
  %c = fadd <4 x float> %res4, %res5
  %d = fadd <4 x float> %res6, %res7
  %e = fadd <4 x float> %res8, %a

  %bc = fadd <4 x float> %b, %c
  %de = fadd <4 x float> %d, %e

  %bcde = fadd <4 x float> %bc, %de

  call void @llvm.R600.store.swizzle(<4 x float> %bcde, i32 0, i32 1)
  ret void
}

declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
