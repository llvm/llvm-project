;RUN: llc < %s -march=r600 -mcpu=cayman -stress-sched -verify-misched -verify-machineinstrs
;REQUIRES: asserts

define void @main() {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(9)* null
  %1 = extractelement <4 x float> %0, i32 3
  %2 = fptosi float %1 to i32
  %3 = bitcast i32 %2 to float
  %4 = bitcast float %3 to i32
  %5 = sdiv i32 %4, 4
  %6 = bitcast i32 %5 to float
  %7 = bitcast float %6 to i32
  %8 = mul i32 %7, 4
  %9 = bitcast i32 %8 to float
  %10 = bitcast float %9 to i32
  %11 = sub i32 0, %10
  %12 = bitcast i32 %11 to float
  %13 = bitcast float %3 to i32
  %14 = bitcast float %12 to i32
  %15 = add i32 %13, %14
  %16 = bitcast i32 %15 to float
  %17 = load <4 x float>, <4 x float> addrspace(9)* null
  %18 = extractelement <4 x float> %17, i32 0
  %19 = load <4 x float>, <4 x float> addrspace(9)* null
  %20 = extractelement <4 x float> %19, i32 1
  %21 = load <4 x float>, <4 x float> addrspace(9)* null
  %22 = extractelement <4 x float> %21, i32 2
  br label %LOOP

LOOP:                                             ; preds = %IF31, %main_body
  %temp12.0 = phi float [ 0.000000e+00, %main_body ], [ %47, %IF31 ]
  %temp6.0 = phi float [ %22, %main_body ], [ %temp6.1, %IF31 ]
  %temp5.0 = phi float [ %20, %main_body ], [ %temp5.1, %IF31 ]
  %temp4.0 = phi float [ %18, %main_body ], [ %temp4.1, %IF31 ]
  %23 = bitcast float %temp12.0 to i32
  %24 = bitcast float %6 to i32
  %25 = icmp sge i32 %23, %24
  %26 = sext i1 %25 to i32
  %27 = bitcast i32 %26 to float
  %28 = bitcast float %27 to i32
  %29 = icmp ne i32 %28, 0
  br i1 %29, label %IF, label %LOOP29

IF:                                               ; preds = %LOOP
  %30 = call float @llvm.AMDGPU.clamp.f32(float %temp4.0, float 0.000000e+00, float 1.000000e+00)
  %31 = call float @llvm.AMDGPU.clamp.f32(float %temp5.0, float 0.000000e+00, float 1.000000e+00)
  %32 = call float @llvm.AMDGPU.clamp.f32(float %temp6.0, float 0.000000e+00, float 1.000000e+00)
  %33 = call float @llvm.AMDGPU.clamp.f32(float 1.000000e+00, float 0.000000e+00, float 1.000000e+00)
  %34 = insertelement <4 x float> undef, float %30, i32 0
  %35 = insertelement <4 x float> %34, float %31, i32 1
  %36 = insertelement <4 x float> %35, float %32, i32 2
  %37 = insertelement <4 x float> %36, float %33, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %37, i32 0, i32 0)
  ret void

LOOP29:                                           ; preds = %LOOP, %ENDIF30
  %temp6.1 = phi float [ %temp4.1, %ENDIF30 ], [ %temp6.0, %LOOP ]
  %temp5.1 = phi float [ %temp6.1, %ENDIF30 ], [ %temp5.0, %LOOP ]
  %temp4.1 = phi float [ %temp5.1, %ENDIF30 ], [ %temp4.0, %LOOP ]
  %temp20.0 = phi float [ %50, %ENDIF30 ], [ 0.000000e+00, %LOOP ]
  %38 = bitcast float %temp20.0 to i32
  %39 = bitcast float %16 to i32
  %40 = icmp sge i32 %38, %39
  %41 = sext i1 %40 to i32
  %42 = bitcast i32 %41 to float
  %43 = bitcast float %42 to i32
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %IF31, label %ENDIF30

IF31:                                             ; preds = %LOOP29
  %45 = bitcast float %temp12.0 to i32
  %46 = add i32 %45, 1
  %47 = bitcast i32 %46 to float
  br label %LOOP

ENDIF30:                                          ; preds = %LOOP29
  %48 = bitcast float %temp20.0 to i32
  %49 = add i32 %48, 1
  %50 = bitcast i32 %49 to float
  br label %LOOP29
}

declare float @llvm.AMDGPU.clamp.f32(float, float, float) #0

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { readnone }
