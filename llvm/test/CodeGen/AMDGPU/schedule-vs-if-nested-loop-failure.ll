; RUN: llc -O0 -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -O0 -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare void @llvm.amdgcn.s.barrier() nounwind convergent

; GCN-LABEL: {{^}}main:
define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 2
  %2 = fcmp ult float %0, 0.000000e+00
  %3 = select i1 %2, float 1.000000e+00, float 0.000000e+00
  %4 = fsub float -0.000000e+00, %3
  %5 = fptosi float %4 to i32
  %6 = bitcast i32 %5 to float
  %7 = bitcast float %6 to i32
  %8 = icmp ne i32 %7, 0
  br i1 %8, label %LOOP, label %ENDIF

Flow1:                                            ; preds = %ENDIF19, %ENDIF16
  %9 = phi float [ %115, %ENDIF19 ], [ undef, %ENDIF16 ]
  %10 = phi float [ %114, %ENDIF19 ], [ undef, %ENDIF16 ]
  %11 = phi float [ %113, %ENDIF19 ], [ undef, %ENDIF16 ]
  %12 = phi float [ %112, %ENDIF19 ], [ undef, %ENDIF16 ]
  %13 = phi float [ %111, %ENDIF19 ], [ undef, %ENDIF16 ]
  %14 = phi i1 [ false, %ENDIF19 ], [ true, %ENDIF16 ]
  br label %Flow

Flow2:                                            ; preds = %Flow
  br label %ENDIF

ENDIF:                                            ; preds = %main_body, %Flow2
  %temp.0 = phi float [ 0.000000e+00, %main_body ], [ %104, %Flow2 ]
  %temp1.0 = phi float [ 1.000000e+00, %main_body ], [ %103, %Flow2 ]
  %temp2.0 = phi float [ 0.000000e+00, %main_body ], [ %102, %Flow2 ]
  %temp3.0 = phi float [ 0.000000e+00, %main_body ], [ %101, %Flow2 ]
  %15 = extractelement <4 x float> %reg1, i32 1
  %16 = extractelement <4 x float> %reg1, i32 3
  %17 = load <4 x float>, ptr addrspace(4) null
  %18 = extractelement <4 x float> %17, i32 0
  %19 = fmul float %18, %0
  %20 = load <4 x float>, ptr addrspace(4) null
  %21 = extractelement <4 x float> %20, i32 1
  %22 = fmul float %21, %0
  %23 = load <4 x float>, ptr addrspace(4) null
  %24 = extractelement <4 x float> %23, i32 2
  %25 = fmul float %24, %0
  %26 = load <4 x float>, ptr addrspace(4) null
  %27 = extractelement <4 x float> %26, i32 3
  %28 = fmul float %27, %0
  %29 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 1)
  %30 = extractelement <4 x float> %29, i32 0
  %31 = fmul float %30, %15
  %32 = fadd float %31, %19
  %33 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 1)
  %34 = extractelement <4 x float> %33, i32 1
  %35 = fmul float %34, %15
  %36 = fadd float %35, %22
  %37 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 1)
  %38 = extractelement <4 x float> %37, i32 2
  %39 = fmul float %38, %15
  %40 = fadd float %39, %25
  %41 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 1)
  %42 = extractelement <4 x float> %41, i32 3
  %43 = fmul float %42, %15
  %44 = fadd float %43, %28
  %45 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 2)
  %46 = extractelement <4 x float> %45, i32 0
  %47 = fmul float %46, %1
  %48 = fadd float %47, %32
  %49 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 2)
  %50 = extractelement <4 x float> %49, i32 1
  %51 = fmul float %50, %1
  %52 = fadd float %51, %36
  %53 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 2)
  %54 = extractelement <4 x float> %53, i32 2
  %55 = fmul float %54, %1
  %56 = fadd float %55, %40
  %57 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 2)
  %58 = extractelement <4 x float> %57, i32 3
  %59 = fmul float %58, %1
  %60 = fadd float %59, %44
  %61 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 3)
  %62 = extractelement <4 x float> %61, i32 0
  %63 = fmul float %62, %16
  %64 = fadd float %63, %48
  %65 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 3)
  %66 = extractelement <4 x float> %65, i32 1
  %67 = fmul float %66, %16
  %68 = fadd float %67, %52
  %69 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 3)
  %70 = extractelement <4 x float> %69, i32 2
  %71 = fmul float %70, %16
  %72 = fadd float %71, %56
  %73 = load <4 x float>, ptr addrspace(4) getelementptr ([1024 x <4 x float>], ptr addrspace(4) null, i64 0, i32 3)
  %74 = extractelement <4 x float> %73, i32 3
  %75 = fmul float %74, %16
  %76 = fadd float %75, %60
  %77 = insertelement <4 x float> undef, float %64, i32 0
  %78 = insertelement <4 x float> %77, float %68, i32 1
  %79 = insertelement <4 x float> %78, float %72, i32 2
  %80 = insertelement <4 x float> %79, float %76, i32 3
  call void @llvm.amdgcn.s.barrier()
  %81 = insertelement <4 x float> undef, float %temp.0, i32 0
  %82 = insertelement <4 x float> %81, float %temp1.0, i32 1
  %83 = insertelement <4 x float> %82, float %temp2.0, i32 2
  %84 = insertelement <4 x float> %83, float %temp3.0, i32 3
  call void @llvm.amdgcn.s.barrier()
  ret void

LOOP:                                             ; preds = %main_body, %Flow
  %temp.1 = phi float [ %109, %Flow ], [ 0.000000e+00, %main_body ]
  %temp1.1 = phi float [ %108, %Flow ], [ 1.000000e+00, %main_body ]
  %temp2.1 = phi float [ %107, %Flow ], [ 0.000000e+00, %main_body ]
  %temp3.1 = phi float [ %106, %Flow ], [ 0.000000e+00, %main_body ]
  %temp4.0 = phi float [ %105, %Flow ], [ -2.000000e+00, %main_body ]
  %85 = fcmp uge float %temp4.0, %0
  %86 = select i1 %85, float 1.000000e+00, float 0.000000e+00
  %87 = fsub float -0.000000e+00, %86
  %88 = fptosi float %87 to i32
  %89 = bitcast i32 %88 to float
  %90 = bitcast float %89 to i32
  %91 = icmp ne i32 %90, 0
  %92 = xor i1 %91, true
  br i1 %92, label %ENDIF16, label %Flow

ENDIF16:                                          ; preds = %LOOP
  %93 = fcmp une float %1, %temp4.0
  %94 = select i1 %93, float 1.000000e+00, float 0.000000e+00
  %95 = fsub float -0.000000e+00, %94
  %96 = fptosi float %95 to i32
  %97 = bitcast i32 %96 to float
  %98 = bitcast float %97 to i32
  %99 = icmp ne i32 %98, 0
  %100 = xor i1 %99, true
  br i1 %100, label %ENDIF19, label %Flow1

Flow:                                             ; preds = %Flow1, %LOOP
  %101 = phi float [ %temp3.1, %Flow1 ], [ %temp3.1, %LOOP ]
  %102 = phi float [ %temp2.1, %Flow1 ], [ %temp2.1, %LOOP ]
  %103 = phi float [ %temp1.1, %Flow1 ], [ %temp1.1, %LOOP ]
  %104 = phi float [ %temp.1, %Flow1 ], [ %temp.1, %LOOP ]
  %105 = phi float [ %9, %Flow1 ], [ undef, %LOOP ]
  %106 = phi float [ %10, %Flow1 ], [ undef, %LOOP ]
  %107 = phi float [ %11, %Flow1 ], [ undef, %LOOP ]
  %108 = phi float [ %12, %Flow1 ], [ undef, %LOOP ]
  %109 = phi float [ %13, %Flow1 ], [ undef, %LOOP ]
  %110 = phi i1 [ %14, %Flow1 ], [ true, %LOOP ]
  br i1 %110, label %Flow2, label %LOOP

ENDIF19:                                          ; preds = %ENDIF16
  %111 = fadd float %temp.1, 1.000000e+00
  %112 = fadd float %temp1.1, 0.000000e+00
  %113 = fadd float %temp2.1, 0.000000e+00
  %114 = fadd float %temp3.1, 0.000000e+00
  %115 = fadd float %temp4.0, 1.000000e+00
  br label %Flow1
}
