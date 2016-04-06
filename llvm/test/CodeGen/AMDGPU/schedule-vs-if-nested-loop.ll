;RUN: llc < %s -march=r600 -mcpu=cayman -stress-sched -verify-misched
;REQUIRES: asserts

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = fcmp ult float %0, 0.000000e+00
  %5 = select i1 %4, float 1.000000e+00, float 0.000000e+00
  %6 = fsub float -0.000000e+00, %5
  %7 = fptosi float %6 to i32
  %8 = bitcast i32 %7 to float
  %9 = bitcast float %8 to i32
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %LOOP, label %ENDIF

ENDIF:                                            ; preds = %ENDIF16, %LOOP, %main_body
  %temp.0 = phi float [ 0.000000e+00, %main_body ], [ %temp.1, %LOOP ], [ %temp.1, %ENDIF16 ]
  %temp1.0 = phi float [ 1.000000e+00, %main_body ], [ %temp1.1, %LOOP ], [ %temp1.1, %ENDIF16 ]
  %temp2.0 = phi float [ 0.000000e+00, %main_body ], [ %temp2.1, %LOOP ], [ %temp2.1, %ENDIF16 ]
  %temp3.0 = phi float [ 0.000000e+00, %main_body ], [ %temp3.1, %LOOP ], [ %temp3.1, %ENDIF16 ]
  %11 = load <4 x float>, <4 x float> addrspace(9)* null
  %12 = extractelement <4 x float> %11, i32 0
  %13 = fmul float %12, %0
  %14 = load <4 x float>, <4 x float> addrspace(9)* null
  %15 = extractelement <4 x float> %14, i32 1
  %16 = fmul float %15, %0
  %17 = load <4 x float>, <4 x float> addrspace(9)* null
  %18 = extractelement <4 x float> %17, i32 2
  %19 = fmul float %18, %0
  %20 = load <4 x float>, <4 x float> addrspace(9)* null
  %21 = extractelement <4 x float> %20, i32 3
  %22 = fmul float %21, %0
  %23 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 1)
  %24 = extractelement <4 x float> %23, i32 0
  %25 = fmul float %24, %1
  %26 = fadd float %25, %13
  %27 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 1)
  %28 = extractelement <4 x float> %27, i32 1
  %29 = fmul float %28, %1
  %30 = fadd float %29, %16
  %31 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 1)
  %32 = extractelement <4 x float> %31, i32 2
  %33 = fmul float %32, %1
  %34 = fadd float %33, %19
  %35 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 1)
  %36 = extractelement <4 x float> %35, i32 3
  %37 = fmul float %36, %1
  %38 = fadd float %37, %22
  %39 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 2)
  %40 = extractelement <4 x float> %39, i32 0
  %41 = fmul float %40, %2
  %42 = fadd float %41, %26
  %43 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 2)
  %44 = extractelement <4 x float> %43, i32 1
  %45 = fmul float %44, %2
  %46 = fadd float %45, %30
  %47 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 2)
  %48 = extractelement <4 x float> %47, i32 2
  %49 = fmul float %48, %2
  %50 = fadd float %49, %34
  %51 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 2)
  %52 = extractelement <4 x float> %51, i32 3
  %53 = fmul float %52, %2
  %54 = fadd float %53, %38
  %55 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 3)
  %56 = extractelement <4 x float> %55, i32 0
  %57 = fmul float %56, %3
  %58 = fadd float %57, %42
  %59 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 3)
  %60 = extractelement <4 x float> %59, i32 1
  %61 = fmul float %60, %3
  %62 = fadd float %61, %46
  %63 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 3)
  %64 = extractelement <4 x float> %63, i32 2
  %65 = fmul float %64, %3
  %66 = fadd float %65, %50
  %67 = load <4 x float>, <4 x float> addrspace(9)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(9)* null, i64 0, i32 3)
  %68 = extractelement <4 x float> %67, i32 3
  %69 = fmul float %68, %3
  %70 = fadd float %69, %54
  %71 = insertelement <4 x float> undef, float %58, i32 0
  %72 = insertelement <4 x float> %71, float %62, i32 1
  %73 = insertelement <4 x float> %72, float %66, i32 2
  %74 = insertelement <4 x float> %73, float %70, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %74, i32 60, i32 1)
  %75 = insertelement <4 x float> undef, float %temp.0, i32 0
  %76 = insertelement <4 x float> %75, float %temp1.0, i32 1
  %77 = insertelement <4 x float> %76, float %temp2.0, i32 2
  %78 = insertelement <4 x float> %77, float %temp3.0, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %78, i32 0, i32 2)
  ret void

LOOP:                                             ; preds = %main_body, %ENDIF19
  %temp.1 = phi float [ %93, %ENDIF19 ], [ 0.000000e+00, %main_body ]
  %temp1.1 = phi float [ %94, %ENDIF19 ], [ 1.000000e+00, %main_body ]
  %temp2.1 = phi float [ %95, %ENDIF19 ], [ 0.000000e+00, %main_body ]
  %temp3.1 = phi float [ %96, %ENDIF19 ], [ 0.000000e+00, %main_body ]
  %temp4.0 = phi float [ %97, %ENDIF19 ], [ -2.000000e+00, %main_body ]
  %79 = fcmp uge float %temp4.0, %0
  %80 = select i1 %79, float 1.000000e+00, float 0.000000e+00
  %81 = fsub float -0.000000e+00, %80
  %82 = fptosi float %81 to i32
  %83 = bitcast i32 %82 to float
  %84 = bitcast float %83 to i32
  %85 = icmp ne i32 %84, 0
  br i1 %85, label %ENDIF, label %ENDIF16

ENDIF16:                                          ; preds = %LOOP
  %86 = fcmp une float %2, %temp4.0
  %87 = select i1 %86, float 1.000000e+00, float 0.000000e+00
  %88 = fsub float -0.000000e+00, %87
  %89 = fptosi float %88 to i32
  %90 = bitcast i32 %89 to float
  %91 = bitcast float %90 to i32
  %92 = icmp ne i32 %91, 0
  br i1 %92, label %ENDIF, label %ENDIF19

ENDIF19:                                          ; preds = %ENDIF16
  %93 = fadd float %temp.1, 1.000000e+00
  %94 = fadd float %temp1.1, 0.000000e+00
  %95 = fadd float %temp2.1, 0.000000e+00
  %96 = fadd float %temp3.1, 0.000000e+00
  %97 = fadd float %temp4.0, 1.000000e+00
  br label %LOOP
}

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
