; RUN: llc < %s -march=r600 -mcpu=rv710 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=rv730 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=rv770 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=cedar | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=sumo | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=juniper | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=cypress | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=barts | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=turks | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=caicos | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s

; r700+ supports 16 fetches in a clause
; CHECK: {{^}}fetch_limits_r700:
; CHECK: Fetch clause
; CHECK: Fetch clause

define amdgpu_ps void @fetch_limits_r700() {
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
  %9 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %10 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 10)
  %11 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 11)
  %12 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 12)
  %13 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 13)
  %14 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 14)
  %15 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 15)
  %16 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 16)
  %res0 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %0, i32 0, i32 0, i32 1)
  %res1 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %1, i32 0, i32 0, i32 1)
  %res2 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %2, i32 0, i32 0, i32 1)
  %res3 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %3, i32 0, i32 0, i32 1)
  %res4 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %4, i32 0, i32 0, i32 1)
  %res5 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %5, i32 0, i32 0, i32 1)
  %res6 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %6, i32 0, i32 0, i32 1)
  %res7 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %7, i32 0, i32 0, i32 1)
  %res8 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 0, i32 0, i32 1)
  %res9 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %9, i32 0, i32 0, i32 1)
  %res10 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %10, i32 0, i32 0, i32 1)
  %res11 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %11, i32 0, i32 0, i32 1)
  %res12 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %12, i32 0, i32 0, i32 1)
  %res13 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %13, i32 0, i32 0, i32 1)
  %res14 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %14, i32 0, i32 0, i32 1)
  %res15 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %15, i32 0, i32 0, i32 1)
  %res16 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %16, i32 0, i32 0, i32 1)
  %a = fadd <4 x float> %res0, %res1
  %b = fadd <4 x float> %res2, %res3
  %c = fadd <4 x float> %res4, %res5
  %d = fadd <4 x float> %res6, %res7
  %e = fadd <4 x float> %res8, %res9
  %f = fadd <4 x float> %res10, %res11
  %g = fadd <4 x float> %res12, %res13
  %h = fadd <4 x float> %res14, %res15
  %i = fadd <4 x float> %res16, %a

  %bc = fadd <4 x float> %b, %c
  %de = fadd <4 x float> %d, %e
  %fg = fadd <4 x float> %f, %g
  %hi = fadd <4 x float> %h, %i

  %bcde = fadd <4 x float> %bc, %de
  %fghi = fadd <4 x float> %fg, %hi

  %bcdefghi = fadd <4 x float> %bcde, %fghi
  call void @llvm.R600.store.swizzle(<4 x float> %bcdefghi, i32 0, i32 1)
  ret void
}

declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
