;RUN: llc < %s -march=r600 -mcpu=cayman -stress-sched -verify-misched -verify-machineinstrs
;REQUIRES: asserts

define amdgpu_kernel void @main() {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %1 = extractelement <4 x float> %0, i32 0
  %2 = fadd float 1.000000e+03, %1
  %3 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %4 = extractelement <4 x float> %3, i32 0
  %5 = bitcast float %4 to i32
  %6 = icmp eq i32 %5, 0
  %7 = sext i1 %6 to i32
  %8 = bitcast i32 %7 to float
  %9 = bitcast float %8 to i32
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %IF, label %ELSE

IF:                                               ; preds = %main_body
  %11 = call float @fabsf(float %2)
  %12 = fcmp ueq float %11, 0x7FF0000000000000
  %13 = select i1 %12, float 1.000000e+00, float 0.000000e+00
  %14 = fsub float -0.000000e+00, %13
  %15 = fptosi float %14 to i32
  %16 = bitcast i32 %15 to float
  %17 = bitcast float %16 to i32
  %18 = icmp ne i32 %17, 0
  %. = select i1 %18, float 0x36A0000000000000, float 0.000000e+00
  %19 = fcmp une float %2, %2
  %20 = select i1 %19, float 1.000000e+00, float 0.000000e+00
  %21 = fsub float -0.000000e+00, %20
  %22 = fptosi float %21 to i32
  %23 = bitcast i32 %22 to float
  %24 = bitcast float %23 to i32
  %25 = icmp ne i32 %24, 0
  %temp8.0 = select i1 %25, float 0x36A0000000000000, float 0.000000e+00
  %26 = bitcast float %. to i32
  %27 = sitofp i32 %26 to float
  %28 = bitcast float %temp8.0 to i32
  %29 = sitofp i32 %28 to float
  %30 = fcmp ugt float %2, 0.000000e+00
  %31 = select i1 %30, float 1.000000e+00, float %2
  %32 = fcmp uge float %31, 0.000000e+00
  %33 = select i1 %32, float %31, float -1.000000e+00
  %34 = fadd float %33, 1.000000e+00
  %35 = fmul float %34, 5.000000e-01
  br label %ENDIF

ELSE:                                             ; preds = %main_body
  %36 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %37 = extractelement <4 x float> %36, i32 0
  %38 = bitcast float %37 to i32
  %39 = icmp eq i32 %38, 1
  %40 = sext i1 %39 to i32
  %41 = bitcast i32 %40 to float
  %42 = bitcast float %41 to i32
  %43 = icmp ne i32 %42, 0
  br i1 %43, label %IF23, label %ENDIF

ENDIF:                                            ; preds = %IF23, %ELSE, %IF
  %temp4.0 = phi float [ %2, %IF ], [ %56, %IF23 ], [ 0.000000e+00, %ELSE ]
  %temp5.0 = phi float [ %27, %IF ], [ %60, %IF23 ], [ 0.000000e+00, %ELSE ]
  %temp6.0 = phi float [ %29, %IF ], [ 0.000000e+00, %ELSE ], [ 0.000000e+00, %IF23 ]
  %temp7.0 = phi float [ %35, %IF ], [ 0.000000e+00, %ELSE ], [ 0.000000e+00, %IF23 ]
  %44 = insertelement <4 x float> undef, float %temp4.0, i32 0
  %45 = insertelement <4 x float> %44, float %temp5.0, i32 1
  %46 = insertelement <4 x float> %45, float %temp6.0, i32 2
  %47 = insertelement <4 x float> %46, float %temp7.0, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %47, i32 0, i32 0)
  ret void

IF23:                                             ; preds = %ELSE
  %48 = fcmp ult float 0.000000e+00, %2
  %49 = select i1 %48, float 1.000000e+00, float 0.000000e+00
  %50 = fsub float -0.000000e+00, %49
  %51 = fptosi float %50 to i32
  %52 = bitcast i32 %51 to float
  %53 = bitcast float %52 to i32
  %54 = icmp ne i32 %53, 0
  %.28 = select i1 %54, float 0x36A0000000000000, float 0.000000e+00
  %55 = bitcast float %.28 to i32
  %56 = sitofp i32 %55 to float
  %57 = load <4 x float>, <4 x float> addrspace(8)* null
  %58 = extractelement <4 x float> %57, i32 0
  %59 = fsub float -0.000000e+00, %58
  %60 = fadd float %2, %59
  br label %ENDIF
}

declare float @fabsf(float) #0

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { readonly }
