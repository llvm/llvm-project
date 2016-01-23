;RUN: llc < %s -march=r600 -mcpu=cayman -stress-sched -verify-misched -verify-machineinstrs
;REQUIRES: asserts

define void @main() {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(9)* null
  %1 = extractelement <4 x float> %0, i32 3
  %2 = fptosi float %1 to i32
  %3 = bitcast i32 %2 to float
  %4 = load <4 x float>, <4 x float> addrspace(9)* null
  %5 = extractelement <4 x float> %4, i32 0
  %6 = load <4 x float>, <4 x float> addrspace(9)* null
  %7 = extractelement <4 x float> %6, i32 1
  %8 = load <4 x float>, <4 x float> addrspace(9)* null
  %9 = extractelement <4 x float> %8, i32 2
  br label %LOOP

LOOP:                                             ; preds = %ENDIF, %main_body
  %temp4.0 = phi float [ %5, %main_body ], [ %temp5.0, %ENDIF ]
  %temp5.0 = phi float [ %7, %main_body ], [ %temp6.0, %ENDIF ]
  %temp6.0 = phi float [ %9, %main_body ], [ %temp4.0, %ENDIF ]
  %temp8.0 = phi float [ 0.000000e+00, %main_body ], [ %27, %ENDIF ]
  %10 = bitcast float %temp8.0 to i32
  %11 = bitcast float %3 to i32
  %12 = icmp sge i32 %10, %11
  %13 = sext i1 %12 to i32
  %14 = bitcast i32 %13 to float
  %15 = bitcast float %14 to i32
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %IF, label %ENDIF

IF:                                               ; preds = %LOOP
  %17 = call float @llvm.AMDGPU.clamp.f32(float %temp4.0, float 0.000000e+00, float 1.000000e+00)
  %18 = call float @llvm.AMDGPU.clamp.f32(float %temp5.0, float 0.000000e+00, float 1.000000e+00)
  %19 = call float @llvm.AMDGPU.clamp.f32(float %temp6.0, float 0.000000e+00, float 1.000000e+00)
  %20 = call float @llvm.AMDGPU.clamp.f32(float 1.000000e+00, float 0.000000e+00, float 1.000000e+00)
  %21 = insertelement <4 x float> undef, float %17, i32 0
  %22 = insertelement <4 x float> %21, float %18, i32 1
  %23 = insertelement <4 x float> %22, float %19, i32 2
  %24 = insertelement <4 x float> %23, float %20, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %24, i32 0, i32 0)
  ret void

ENDIF:                                            ; preds = %LOOP
  %25 = bitcast float %temp8.0 to i32
  %26 = add i32 %25, 1
  %27 = bitcast i32 %26 to float
  br label %LOOP
}

declare float @llvm.AMDGPU.clamp.f32(float, float, float) #0

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { readnone }
