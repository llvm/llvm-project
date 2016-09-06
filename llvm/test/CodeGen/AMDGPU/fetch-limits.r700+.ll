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
  %17 = shufflevector <4 x float> %0, <4 x float> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %18 = call <4 x float> @llvm.r600.tex(<4 x float> %17, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %19 = shufflevector <4 x float> %1, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %20 = call <4 x float> @llvm.r600.tex(<4 x float> %19, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %21 = shufflevector <4 x float> %2, <4 x float> %2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %22 = call <4 x float> @llvm.r600.tex(<4 x float> %21, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %23 = shufflevector <4 x float> %3, <4 x float> %3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %24 = call <4 x float> @llvm.r600.tex(<4 x float> %23, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %25 = shufflevector <4 x float> %4, <4 x float> %4, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %26 = call <4 x float> @llvm.r600.tex(<4 x float> %25, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %27 = shufflevector <4 x float> %5, <4 x float> %5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %28 = call <4 x float> @llvm.r600.tex(<4 x float> %27, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %29 = shufflevector <4 x float> %6, <4 x float> %6, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %30 = call <4 x float> @llvm.r600.tex(<4 x float> %29, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %31 = shufflevector <4 x float> %7, <4 x float> %7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %32 = call <4 x float> @llvm.r600.tex(<4 x float> %31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %33 = shufflevector <4 x float> %8, <4 x float> %8, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %34 = call <4 x float> @llvm.r600.tex(<4 x float> %33, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %35 = shufflevector <4 x float> %9, <4 x float> %9, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %36 = call <4 x float> @llvm.r600.tex(<4 x float> %35, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %37 = shufflevector <4 x float> %10, <4 x float> %10, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %38 = call <4 x float> @llvm.r600.tex(<4 x float> %37, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %39 = shufflevector <4 x float> %11, <4 x float> %11, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %40 = call <4 x float> @llvm.r600.tex(<4 x float> %39, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %41 = shufflevector <4 x float> %12, <4 x float> %12, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %42 = call <4 x float> @llvm.r600.tex(<4 x float> %41, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %43 = shufflevector <4 x float> %13, <4 x float> %13, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %44 = call <4 x float> @llvm.r600.tex(<4 x float> %43, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %45 = shufflevector <4 x float> %14, <4 x float> %14, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %46 = call <4 x float> @llvm.r600.tex(<4 x float> %45, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %47 = shufflevector <4 x float> %15, <4 x float> %15, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %48 = call <4 x float> @llvm.r600.tex(<4 x float> %47, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %49 = shufflevector <4 x float> %16, <4 x float> %16, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %50 = call <4 x float> @llvm.r600.tex(<4 x float> %49, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %a = fadd <4 x float> %18, %20
  %b = fadd <4 x float> %22, %24
  %c = fadd <4 x float> %26, %28
  %d = fadd <4 x float> %30, %32
  %e = fadd <4 x float> %34, %36
  %f = fadd <4 x float> %38, %40
  %g = fadd <4 x float> %42, %44
  %h = fadd <4 x float> %46, %48
  %i = fadd <4 x float> %50, %a
  %bc = fadd <4 x float> %b, %c
  %de = fadd <4 x float> %d, %e
  %fg = fadd <4 x float> %f, %g
  %hi = fadd <4 x float> %h, %i
  %bcde = fadd <4 x float> %bc, %de
  %fghi = fadd <4 x float> %fg, %hi
  %bcdefghi = fadd <4 x float> %bcde, %fghi
  call void @llvm.r600.store.swizzle(<4 x float> %bcdefghi, i32 0, i32 1)
  ret void
}

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

; Function Attrs: readnone
declare <4 x float> @llvm.r600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { readnone }
