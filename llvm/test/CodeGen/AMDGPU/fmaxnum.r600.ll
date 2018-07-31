; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -enable-var-scope -check-prefix=EG %s

; EG-LABEL: {{^}}test_fmax_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MAX_DX10 {{.*}}[[OUT]]
define amdgpu_kernel void @test_fmax_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float %b)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}test_fmax_v2f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+]]
; EG: MAX_DX10 {{.*}}[[OUT]]
; EG: MAX_DX10 {{.*}}[[OUT]]
define amdgpu_kernel void @test_fmax_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) #0 {
  %val = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %a, <2 x float> %b)
  store <2 x float> %val, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; EG-LABEL: {{^}}test_fmax_v4f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+]]
; EG: MAX_DX10 {{.*}}[[OUT]]
; EG: MAX_DX10 {{.*}}[[OUT]]
; EG: MAX_DX10 {{.*}}[[OUT]]
; EG: MAX_DX10 {{.*}}[[OUT]]
define amdgpu_kernel void @test_fmax_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b) #0 {
  %val = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %a, <4 x float> %b)
  store <4 x float> %val, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; EG-LABEL: {{^}}test_fmax_v8f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT1:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT2:T[0-9]+]]
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].X
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].Y
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].Z
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].W
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].X
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].Y
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].Z
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].W
define amdgpu_kernel void @test_fmax_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %a, <8 x float> %b) #0 {
  %val = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %a, <8 x float> %b)
  store <8 x float> %val, <8 x float> addrspace(1)* %out, align 32
  ret void
}

; EG-LABEL: {{^}}test_fmax_v16f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT1:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT2:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT3:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT4:T[0-9]+]]
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].X
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].Y
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].Z
; EG-DAG: MAX_DX10 {{.*}}[[OUT1]].W
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].X
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].Y
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].Z
; EG-DAG: MAX_DX10 {{.*}}[[OUT2]].W
; EG-DAG: MAX_DX10 {{.*}}[[OUT3]].X
; EG-DAG: MAX_DX10 {{.*}}[[OUT3]].Y
; EG-DAG: MAX_DX10 {{.*}}[[OUT3]].Z
; EG-DAG: MAX_DX10 {{.*}}[[OUT3]].W
; EG-DAG: MAX_DX10 {{.*}}[[OUT4]].X
; EG-DAG: MAX_DX10 {{.*}}[[OUT4]].Y
; EG-DAG: MAX_DX10 {{.*}}[[OUT4]].Z
; EG-DAG: MAX_DX10 {{.*}}[[OUT4]].W
define amdgpu_kernel void @test_fmax_v16f32(<16 x float> addrspace(1)* %out, <16 x float> %a, <16 x float> %b) #0 {
  %val = call <16 x float> @llvm.maxnum.v16f32(<16 x float> %a, <16 x float> %b)
  store <16 x float> %val, <16 x float> addrspace(1)* %out, align 64
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float 1.0, float 2.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_nan_nan:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
; EG: 2143289344(nan)
define amdgpu_kernel void @constant_fold_fmax_f32_nan_nan(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float 0x7FF8000000000000)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_val_nan:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32_val_nan(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float 1.0, float 0x7FF8000000000000)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_nan_val:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32_nan_val(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0x7FF8000000000000, float 1.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_p0_p0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32_p0_p0(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0.0, float 0.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_p0_n0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32_p0_n0(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float 0.0, float -0.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_n0_p0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32_n0_p0(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float -0.0, float 0.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}constant_fold_fmax_f32_n0_n0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MAX_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define amdgpu_kernel void @constant_fold_fmax_f32_n0_n0(float addrspace(1)* %out) #0 {
  %val = call float @llvm.maxnum.f32(float -0.0, float -0.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}fmax_var_immediate_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MAX_DX10 * [[OUT]]
define amdgpu_kernel void @fmax_var_immediate_f32(float addrspace(1)* %out, float %a) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float 2.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}fmax_immediate_var_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MAX_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define amdgpu_kernel void @fmax_immediate_var_f32(float addrspace(1)* %out, float %a) #0 {
  %val = call float @llvm.maxnum.f32(float 2.0, float %a)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}fmax_var_literal_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MAX_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define amdgpu_kernel void @fmax_var_literal_f32(float addrspace(1)* %out, float %a) #0 {
  %val = call float @llvm.maxnum.f32(float %a, float 99.0)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; EG-LABEL: {{^}}fmax_literal_var_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MAX_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define amdgpu_kernel void @fmax_literal_var_f32(float addrspace(1)* %out, float %a) #0 {
  %val = call float @llvm.maxnum.f32(float 99.0, float %a)
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

declare float @llvm.maxnum.f32(float, float) #1
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>) #1
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) #1
declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>) #1
declare <16 x float> @llvm.maxnum.v16f32(<16 x float>, <16 x float>) #1
declare double @llvm.maxnum.f64(double, double)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
