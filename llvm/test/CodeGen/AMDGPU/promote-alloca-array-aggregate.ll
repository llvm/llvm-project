; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-promote-alloca < %s | FileCheck --check-prefix=OPT %s

; Make sure that array alloca loaded and stored as multi-element aggregates are handled correctly
; Strictly the promote-alloca pass shouldn't have to deal with this case as it is non-canonical, but
; the pass should handle it gracefully if it is
; The checks look for lines that previously caused issues in PromoteAlloca (non-canonical). Opt
; should now leave these unchanged

; OPT-LABEL: @promote_1d_aggr(
; OPT: store [1 x float] %tmp3, [1 x float]* %f1

%Block = type { [1 x float], i32 }
%gl_PerVertex = type { <4 x float>, float, [1 x float], [1 x float] }

@block = external addrspace(1) global %Block
@pv = external addrspace(1) global %gl_PerVertex

define amdgpu_vs void @promote_1d_aggr() #0 {
  %i = alloca i32
  %f1 = alloca [1 x float]
  %tmp = getelementptr %Block, %Block addrspace(1)* @block, i32 0, i32 1
  %tmp1 = load i32, i32 addrspace(1)* %tmp
  store i32 %tmp1, i32* %i
  %tmp2 = getelementptr %Block, %Block addrspace(1)* @block, i32 0, i32 0
  %tmp3 = load [1 x float], [1 x float] addrspace(1)* %tmp2
  store [1 x float] %tmp3, [1 x float]* %f1
  %tmp4 = load i32, i32* %i
  %tmp5 = getelementptr [1 x float], [1 x float]* %f1, i32 0, i32 %tmp4
  %tmp6 = load float, float* %tmp5
  %tmp7 = alloca <4 x float>
  %tmp8 = load <4 x float>, <4 x float>* %tmp7
  %tmp9 = insertelement <4 x float> %tmp8, float %tmp6, i32 0
  %tmp10 = insertelement <4 x float> %tmp9, float %tmp6, i32 1
  %tmp11 = insertelement <4 x float> %tmp10, float %tmp6, i32 2
  %tmp12 = insertelement <4 x float> %tmp11, float %tmp6, i32 3
  %tmp13 = getelementptr %gl_PerVertex, %gl_PerVertex addrspace(1)* @pv, i32 0, i32 0
  store <4 x float> %tmp12, <4 x float> addrspace(1)* %tmp13
  ret void
}


; OPT-LABEL: @promote_store_aggr(
; OPT: %tmp6 = load [2 x float], [2 x float]* %f1

%Block2 = type { i32, [2 x float] }
@block2 = external addrspace(1) global %Block2

define amdgpu_vs void @promote_store_aggr() #0 {
  %i = alloca i32
  %f1 = alloca [2 x float]
  %tmp = getelementptr %Block2, %Block2 addrspace(1)* @block2, i32 0, i32 0
  %tmp1 = load i32, i32 addrspace(1)* %tmp
  store i32 %tmp1, i32* %i
  %tmp2 = load i32, i32* %i
  %tmp3 = sitofp i32 %tmp2 to float
  %tmp4 = getelementptr [2 x float], [2 x float]* %f1, i32 0, i32 0
  store float %tmp3, float* %tmp4
  %tmp5 = getelementptr [2 x float], [2 x float]* %f1, i32 0, i32 1
  store float 2.000000e+00, float* %tmp5
  %tmp6 = load [2 x float], [2 x float]* %f1
  %tmp7 = getelementptr %Block2, %Block2 addrspace(1)* @block2, i32 0, i32 1
  store [2 x float] %tmp6, [2 x float] addrspace(1)* %tmp7
  %tmp8 = getelementptr %gl_PerVertex, %gl_PerVertex addrspace(1)* @pv, i32 0, i32 0
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> addrspace(1)* %tmp8
  ret void
}

; OPT-LABEL: @promote_load_from_store_aggr(
; OPT: store [2 x float] %tmp3, [2 x float]* %f1

%Block3 = type { [2 x float], i32 }
@block3 = external addrspace(1) global %Block3

define amdgpu_vs void @promote_load_from_store_aggr() #0 {
  %i = alloca i32
  %f1 = alloca [2 x float]
  %tmp = getelementptr %Block3, %Block3 addrspace(1)* @block3, i32 0, i32 1
  %tmp1 = load i32, i32 addrspace(1)* %tmp
  store i32 %tmp1, i32* %i
  %tmp2 = getelementptr %Block3, %Block3 addrspace(1)* @block3, i32 0, i32 0
  %tmp3 = load [2 x float], [2 x float] addrspace(1)* %tmp2
  store [2 x float] %tmp3, [2 x float]* %f1
  %tmp4 = load i32, i32* %i
  %tmp5 = getelementptr [2 x float], [2 x float]* %f1, i32 0, i32 %tmp4
  %tmp6 = load float, float* %tmp5
  %tmp7 = alloca <4 x float>
  %tmp8 = load <4 x float>, <4 x float>* %tmp7
  %tmp9 = insertelement <4 x float> %tmp8, float %tmp6, i32 0
  %tmp10 = insertelement <4 x float> %tmp9, float %tmp6, i32 1
  %tmp11 = insertelement <4 x float> %tmp10, float %tmp6, i32 2
  %tmp12 = insertelement <4 x float> %tmp11, float %tmp6, i32 3
  %tmp13 = getelementptr %gl_PerVertex, %gl_PerVertex addrspace(1)* @pv, i32 0, i32 0
  store <4 x float> %tmp12, <4 x float> addrspace(1)* %tmp13
  ret void
}

; OPT-LABEL: @promote_double_aggr(
; OPT: store [2 x double] %tmp5, [2 x double]* %s

@tmp_g = external addrspace(1) global { [4 x double], <2 x double>, <3 x double>, <4 x double> }
@frag_color = external addrspace(1) global <4 x float>

define amdgpu_ps void @promote_double_aggr() #0 {
  %s = alloca [2 x double]
  %tmp = getelementptr { [4 x double], <2 x double>, <3 x double>, <4 x double> }, { [4 x double], <2 x double>, <3 x double>, <4 x double> } addrspace(1)* @tmp_g, i32 0, i32 0, i32 0
  %tmp1 = load double, double addrspace(1)* %tmp
  %tmp2 = getelementptr { [4 x double], <2 x double>, <3 x double>, <4 x double> }, { [4 x double], <2 x double>, <3 x double>, <4 x double> } addrspace(1)* @tmp_g, i32 0, i32 0, i32 1
  %tmp3 = load double, double addrspace(1)* %tmp2
  %tmp4 = insertvalue [2 x double] undef, double %tmp1, 0
  %tmp5 = insertvalue [2 x double] %tmp4, double %tmp3, 1
  store [2 x double] %tmp5, [2 x double]* %s
  %tmp6 = getelementptr [2 x double], [2 x double]* %s, i32 0, i32 1
  %tmp7 = load double, double* %tmp6
  %tmp8 = getelementptr [2 x double], [2 x double]* %s, i32 0, i32 1
  %tmp9 = load double, double* %tmp8
  %tmp10 = fadd double %tmp7, %tmp9
  %tmp11 = getelementptr [2 x double], [2 x double]* %s, i32 0, i32 0
  store double %tmp10, double* %tmp11
  %tmp12 = getelementptr [2 x double], [2 x double]* %s, i32 0, i32 0
  %tmp13 = load double, double* %tmp12
  %tmp14 = getelementptr [2 x double], [2 x double]* %s, i32 0, i32 1
  %tmp15 = load double, double* %tmp14
  %tmp16 = fadd double %tmp13, %tmp15
  %tmp17 = fptrunc double %tmp16 to float
  %tmp18 = insertelement <4 x float> undef, float %tmp17, i32 0
  %tmp19 = insertelement <4 x float> %tmp18, float %tmp17, i32 1
  %tmp20 = insertelement <4 x float> %tmp19, float %tmp17, i32 2
  %tmp21 = insertelement <4 x float> %tmp20, float %tmp17, i32 3
  store <4 x float> %tmp21, <4 x float> addrspace(1)* @frag_color
  ret void
}
