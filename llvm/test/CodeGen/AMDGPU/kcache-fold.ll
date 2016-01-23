;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: {{^}}main1:
; CHECK: MOV * T{{[0-9]+\.[XYZW], KC0}}
define void @main1() {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(8)* null
  %1 = extractelement <4 x float> %0, i32 0
  %2 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %3 = extractelement <4 x float> %2, i32 0
  %4 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %5 = extractelement <4 x float> %4, i32 0
  %6 = fcmp ogt float %1, 0.000000e+00
  %7 = select i1 %6, float %3, float %5
  %8 = load <4 x float>, <4 x float> addrspace(8)* null
  %9 = extractelement <4 x float> %8, i32 1
  %10 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %11 = extractelement <4 x float> %10, i32 1
  %12 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %13 = extractelement <4 x float> %12, i32 1
  %14 = fcmp ogt float %9, 0.000000e+00
  %15 = select i1 %14, float %11, float %13
  %16 = load <4 x float>, <4 x float> addrspace(8)* null
  %17 = extractelement <4 x float> %16, i32 2
  %18 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %19 = extractelement <4 x float> %18, i32 2
  %20 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %21 = extractelement <4 x float> %20, i32 2
  %22 = fcmp ogt float %17, 0.000000e+00
  %23 = select i1 %22, float %19, float %21
  %24 = load <4 x float>, <4 x float> addrspace(8)* null
  %25 = extractelement <4 x float> %24, i32 3
  %26 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %27 = extractelement <4 x float> %26, i32 3
  %28 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %29 = extractelement <4 x float> %28, i32 3
  %30 = fcmp ogt float %25, 0.000000e+00
  %31 = select i1 %30, float %27, float %29
  %32 = call float @llvm.AMDGPU.clamp.f32(float %7, float 0.000000e+00, float 1.000000e+00)
  %33 = call float @llvm.AMDGPU.clamp.f32(float %15, float 0.000000e+00, float 1.000000e+00)
  %34 = call float @llvm.AMDGPU.clamp.f32(float %23, float 0.000000e+00, float 1.000000e+00)
  %35 = call float @llvm.AMDGPU.clamp.f32(float %31, float 0.000000e+00, float 1.000000e+00)
  %36 = insertelement <4 x float> undef, float %32, i32 0
  %37 = insertelement <4 x float> %36, float %33, i32 1
  %38 = insertelement <4 x float> %37, float %34, i32 2
  %39 = insertelement <4 x float> %38, float %35, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %39, i32 0, i32 0)
  ret void
}

; CHECK: {{^}}main2:
; CHECK-NOT: MOV
define void @main2() {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(8)* null
  %1 = extractelement <4 x float> %0, i32 0
  %2 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %3 = extractelement <4 x float> %2, i32 0
  %4 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %5 = extractelement <4 x float> %4, i32 1
  %6 = fcmp ogt float %1, 0.000000e+00
  %7 = select i1 %6, float %3, float %5
  %8 = load <4 x float>, <4 x float> addrspace(8)* null
  %9 = extractelement <4 x float> %8, i32 1
  %10 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %11 = extractelement <4 x float> %10, i32 0
  %12 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %13 = extractelement <4 x float> %12, i32 1
  %14 = fcmp ogt float %9, 0.000000e+00
  %15 = select i1 %14, float %11, float %13
  %16 = load <4 x float>, <4 x float> addrspace(8)* null
  %17 = extractelement <4 x float> %16, i32 2
  %18 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %19 = extractelement <4 x float> %18, i32 3
  %20 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %21 = extractelement <4 x float> %20, i32 2
  %22 = fcmp ogt float %17, 0.000000e+00
  %23 = select i1 %22, float %19, float %21
  %24 = load <4 x float>, <4 x float> addrspace(8)* null
  %25 = extractelement <4 x float> %24, i32 3
  %26 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %27 = extractelement <4 x float> %26, i32 3
  %28 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %29 = extractelement <4 x float> %28, i32 2
  %30 = fcmp ogt float %25, 0.000000e+00
  %31 = select i1 %30, float %27, float %29
  %32 = call float @llvm.AMDGPU.clamp.f32(float %7, float 0.000000e+00, float 1.000000e+00)
  %33 = call float @llvm.AMDGPU.clamp.f32(float %15, float 0.000000e+00, float 1.000000e+00)
  %34 = call float @llvm.AMDGPU.clamp.f32(float %23, float 0.000000e+00, float 1.000000e+00)
  %35 = call float @llvm.AMDGPU.clamp.f32(float %31, float 0.000000e+00, float 1.000000e+00)
  %36 = insertelement <4 x float> undef, float %32, i32 0
  %37 = insertelement <4 x float> %36, float %33, i32 1
  %38 = insertelement <4 x float> %37, float %34, i32 2
  %39 = insertelement <4 x float> %38, float %35, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %39, i32 0, i32 0)
  ret void
}

declare float @llvm.AMDGPU.clamp.f32(float, float, float) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
