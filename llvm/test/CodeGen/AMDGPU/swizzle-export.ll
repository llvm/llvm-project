; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s

;EG: {{^}}main:
;EG: EXPORT T{{[0-9]+}}.XYXX
;EG: EXPORT T{{[0-9]+}}.ZXXX
;EG: EXPORT T{{[0-9]+}}.XXWX
;EG: EXPORT T{{[0-9]+}}.XXXW

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = load <4 x float>, <4 x float> addrspace(8)* null
  %5 = extractelement <4 x float> %4, i32 1
  %6 = load <4 x float>, <4 x float> addrspace(8)* null
  %7 = extractelement <4 x float> %6, i32 2
  %8 = load <4 x float>, <4 x float> addrspace(8)* null
  %9 = extractelement <4 x float> %8, i32 0
  %10 = fmul float 0.000000e+00, %9
  %11 = load <4 x float>, <4 x float> addrspace(8)* null
  %12 = extractelement <4 x float> %11, i32 0
  %13 = fmul float %5, %12
  %14 = load <4 x float>, <4 x float> addrspace(8)* null
  %15 = extractelement <4 x float> %14, i32 0
  %16 = fmul float 0.000000e+00, %15
  %17 = load <4 x float>, <4 x float> addrspace(8)* null
  %18 = extractelement <4 x float> %17, i32 0
  %19 = fmul float 0.000000e+00, %18
  %20 = load <4 x float>, <4 x float> addrspace(8)* null
  %21 = extractelement <4 x float> %20, i32 0
  %22 = fmul float %7, %21
  %23 = load <4 x float>, <4 x float> addrspace(8)* null
  %24 = extractelement <4 x float> %23, i32 0
  %25 = fmul float 0.000000e+00, %24
  %26 = load <4 x float>, <4 x float> addrspace(8)* null
  %27 = extractelement <4 x float> %26, i32 0
  %28 = fmul float 0.000000e+00, %27
  %29 = load <4 x float>, <4 x float> addrspace(8)* null
  %30 = extractelement <4 x float> %29, i32 0
  %31 = fmul float 0.000000e+00, %30
  %32 = load <4 x float>, <4 x float> addrspace(8)* null
  %33 = extractelement <4 x float> %32, i32 0
  %34 = fmul float 0.000000e+00, %33
  %35 = load <4 x float>, <4 x float> addrspace(8)* null
  %36 = extractelement <4 x float> %35, i32 0
  %37 = fmul float 0.000000e+00, %36
  %38 = load <4 x float>, <4 x float> addrspace(8)* null
  %39 = extractelement <4 x float> %38, i32 0
  %40 = fmul float 1.000000e+00, %39
  %41 = load <4 x float>, <4 x float> addrspace(8)* null
  %42 = extractelement <4 x float> %41, i32 0
  %43 = fmul float 0.000000e+00, %42
  %44 = load <4 x float>, <4 x float> addrspace(8)* null
  %45 = extractelement <4 x float> %44, i32 0
  %46 = fmul float 0.000000e+00, %45
  %47 = load <4 x float>, <4 x float> addrspace(8)* null
  %48 = extractelement <4 x float> %47, i32 0
  %49 = fmul float 0.000000e+00, %48
  %50 = load <4 x float>, <4 x float> addrspace(8)* null
  %51 = extractelement <4 x float> %50, i32 0
  %52 = fmul float 0.000000e+00, %51
  %53 = load <4 x float>, <4 x float> addrspace(8)* null
  %54 = extractelement <4 x float> %53, i32 0
  %55 = fmul float 1.000000e+00, %54
  %56 = insertelement <4 x float> undef, float %0, i32 0
  %57 = insertelement <4 x float> %56, float %1, i32 1
  %58 = insertelement <4 x float> %57, float %2, i32 2
  %59 = insertelement <4 x float> %58, float %3, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %59, i32 60, i32 1)
  %60 = insertelement <4 x float> undef, float %10, i32 0
  %61 = insertelement <4 x float> %60, float %13, i32 1
  %62 = insertelement <4 x float> %61, float %16, i32 2
  %63 = insertelement <4 x float> %62, float %19, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %63, i32 0, i32 2)
  %64 = insertelement <4 x float> undef, float %22, i32 0
  %65 = insertelement <4 x float> %64, float %25, i32 1
  %66 = insertelement <4 x float> %65, float %28, i32 2
  %67 = insertelement <4 x float> %66, float %31, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %67, i32 1, i32 2)
  %68 = insertelement <4 x float> undef, float %34, i32 0
  %69 = insertelement <4 x float> %68, float %37, i32 1
  %70 = insertelement <4 x float> %69, float %40, i32 2
  %71 = insertelement <4 x float> %70, float %43, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %71, i32 2, i32 2)
  %72 = insertelement <4 x float> undef, float %46, i32 0
  %73 = insertelement <4 x float> %72, float %49, i32 1
  %74 = insertelement <4 x float> %73, float %52, i32 2
  %75 = insertelement <4 x float> %74, float %55, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %75, i32 3, i32 2)
  ret void
}

; EG: {{^}}main2:
; EG: T{{[0-9]+}}.XY__
; EG: T{{[0-9]+}}.ZXY0

define amdgpu_vs void @main2(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = fadd float %0, 2.5
  %3 = fmul float %1, 3.5
  %4 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %5 = extractelement <4 x float> %4, i32 0
  %6 = call float @llvm.cos.f32(float %5)
  %7 = load <4 x float>, <4 x float> addrspace(8)* null
  %8 = extractelement <4 x float> %7, i32 0
  %9 = load <4 x float>, <4 x float> addrspace(8)* null
  %10 = extractelement <4 x float> %9, i32 1
  %11 = insertelement <4 x float> undef, float %2, i32 0
  %12 = insertelement <4 x float> %11, float %3, i32 1
  call void @llvm.r600.store.swizzle(<4 x float> %12, i32 60, i32 1)
  %13 = insertelement <4 x float> undef, float %6, i32 0
  %14 = insertelement <4 x float> %13, float %8, i32 1
  %15 = insertelement <4 x float> %14, float %10, i32 2
  %16 = insertelement <4 x float> %15, float 0.000000e+00, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %16, i32 0, i32 2)
  ret void
}

; Function Attrs: nounwind readonly
declare float @llvm.cos.f32(float) #1

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

attributes #1 = { nounwind readonly }
