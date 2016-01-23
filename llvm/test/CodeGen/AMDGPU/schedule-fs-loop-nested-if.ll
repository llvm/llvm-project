;RUN: llc < %s -march=r600 -mcpu=cayman -stress-sched -verify-misched -verify-machineinstrs
;REQUIRES: asserts

define void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1) #1 {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = fcmp ult float %1, 0.000000e+00
  %5 = select i1 %4, float 1.000000e+00, float 0.000000e+00
  %6 = fsub float -0.000000e+00, %5
  %7 = fptosi float %6 to i32
  %8 = bitcast i32 %7 to float
  %9 = fcmp ult float %0, 5.700000e+01
  %10 = select i1 %9, float 1.000000e+00, float 0.000000e+00
  %11 = fsub float -0.000000e+00, %10
  %12 = fptosi float %11 to i32
  %13 = bitcast i32 %12 to float
  %14 = bitcast float %8 to i32
  %15 = bitcast float %13 to i32
  %16 = and i32 %14, %15
  %17 = bitcast i32 %16 to float
  %18 = bitcast float %17 to i32
  %19 = icmp ne i32 %18, 0
  %20 = fcmp ult float %0, 0.000000e+00
  %21 = select i1 %20, float 1.000000e+00, float 0.000000e+00
  %22 = fsub float -0.000000e+00, %21
  %23 = fptosi float %22 to i32
  %24 = bitcast i32 %23 to float
  %25 = bitcast float %24 to i32
  %26 = icmp ne i32 %25, 0
  br i1 %19, label %IF, label %ELSE

IF:                                               ; preds = %main_body
  %. = select i1 %26, float 0.000000e+00, float 1.000000e+00
  %.18 = select i1 %26, float 1.000000e+00, float 0.000000e+00
  br label %ENDIF

ELSE:                                             ; preds = %main_body
  br i1 %26, label %ENDIF, label %ELSE17

ENDIF:                                            ; preds = %ELSE17, %ELSE, %IF
  %temp1.0 = phi float [ %., %IF ], [ %48, %ELSE17 ], [ 0.000000e+00, %ELSE ]
  %temp2.0 = phi float [ 0.000000e+00, %IF ], [ %49, %ELSE17 ], [ 1.000000e+00, %ELSE ]
  %temp.0 = phi float [ %.18, %IF ], [ %47, %ELSE17 ], [ 0.000000e+00, %ELSE ]
  %27 = call float @llvm.AMDGPU.clamp.f32(float %temp.0, float 0.000000e+00, float 1.000000e+00)
  %28 = call float @llvm.AMDGPU.clamp.f32(float %temp1.0, float 0.000000e+00, float 1.000000e+00)
  %29 = call float @llvm.AMDGPU.clamp.f32(float %temp2.0, float 0.000000e+00, float 1.000000e+00)
  %30 = call float @llvm.AMDGPU.clamp.f32(float 1.000000e+00, float 0.000000e+00, float 1.000000e+00)
  %31 = insertelement <4 x float> undef, float %27, i32 0
  %32 = insertelement <4 x float> %31, float %28, i32 1
  %33 = insertelement <4 x float> %32, float %29, i32 2
  %34 = insertelement <4 x float> %33, float %30, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %34, i32 0, i32 0)
  ret void

ELSE17:                                           ; preds = %ELSE
  %35 = fadd float 0.000000e+00, 0x3FC99999A0000000
  %36 = fadd float 0.000000e+00, 0x3FC99999A0000000
  %37 = fadd float 0.000000e+00, 0x3FC99999A0000000
  %38 = fadd float %35, 0x3FC99999A0000000
  %39 = fadd float %36, 0x3FC99999A0000000
  %40 = fadd float %37, 0x3FC99999A0000000
  %41 = fadd float %38, 0x3FC99999A0000000
  %42 = fadd float %39, 0x3FC99999A0000000
  %43 = fadd float %40, 0x3FC99999A0000000
  %44 = fadd float %41, 0x3FC99999A0000000
  %45 = fadd float %42, 0x3FC99999A0000000
  %46 = fadd float %43, 0x3FC99999A0000000
  %47 = fadd float %44, 0x3FC99999A0000000
  %48 = fadd float %45, 0x3FC99999A0000000
  %49 = fadd float %46, 0x3FC99999A0000000
  br label %ENDIF
}

declare float @llvm.AMDGPU.clamp.f32(float, float, float) #0

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { readnone }
attributes #1 = { "ShaderType"="1" }
