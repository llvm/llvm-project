; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.minnum.f32(float, float) #0
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>) #0
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>) #0
declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>) #0
declare <16 x float> @llvm.minnum.v16f32(<16 x float>, <16 x float>) #0

; FUNC-LABEL: @test_fmin_f32
; SI: v_min_f32_e32

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MIN_DX10 {{.*}}[[OUT]]
define void @test_fmin_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %val = call float @llvm.minnum.f32(float %a, float %b) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_v2f32
; SI: v_min_f32_e32
; SI: v_min_f32_e32

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+]]
; EG: MIN_DX10 {{.*}}[[OUT]]
; EG: MIN_DX10 {{.*}}[[OUT]]
define void @test_fmin_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) nounwind {
  %val = call <2 x float> @llvm.minnum.v2f32(<2 x float> %a, <2 x float> %b) #0
  store <2 x float> %val, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_fmin_v4f32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+]]
; EG: MIN_DX10 {{.*}}[[OUT]]
; EG: MIN_DX10 {{.*}}[[OUT]]
; EG: MIN_DX10 {{.*}}[[OUT]]
; EG: MIN_DX10 {{.*}}[[OUT]]
define void @test_fmin_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b) nounwind {
  %val = call <4 x float> @llvm.minnum.v4f32(<4 x float> %a, <4 x float> %b) #0
  store <4 x float> %val, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: @test_fmin_v8f32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT1:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT2:T[0-9]+]]
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].X
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].Y
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].Z
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].W
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].X
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].Y
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].Z
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].W
define void @test_fmin_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %a, <8 x float> %b) nounwind {
  %val = call <8 x float> @llvm.minnum.v8f32(<8 x float> %a, <8 x float> %b) #0
  store <8 x float> %val, <8 x float> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: @test_fmin_v16f32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32
; SI: v_min_f32_e32

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT1:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT2:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT3:T[0-9]+]]
; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT4:T[0-9]+]]
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].X
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].Y
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].Z
; EG-DAG: MIN_DX10 {{.*}}[[OUT1]].W
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].X
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].Y
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].Z
; EG-DAG: MIN_DX10 {{.*}}[[OUT2]].W
; EG-DAG: MIN_DX10 {{.*}}[[OUT3]].X
; EG-DAG: MIN_DX10 {{.*}}[[OUT3]].Y
; EG-DAG: MIN_DX10 {{.*}}[[OUT3]].Z
; EG-DAG: MIN_DX10 {{.*}}[[OUT3]].W
; EG-DAG: MIN_DX10 {{.*}}[[OUT4]].X
; EG-DAG: MIN_DX10 {{.*}}[[OUT4]].Y
; EG-DAG: MIN_DX10 {{.*}}[[OUT4]].Z
; EG-DAG: MIN_DX10 {{.*}}[[OUT4]].W
define void @test_fmin_v16f32(<16 x float> addrspace(1)* %out, <16 x float> %a, <16 x float> %b) nounwind {
  %val = call <16 x float> @llvm.minnum.v16f32(<16 x float> %a, <16 x float> %b) #0
  store <16 x float> %val, <16 x float> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32
; SI-NOT: v_min_f32_e32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float 1.0, float 2.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_nan_nan
; SI-NOT: v_min_f32_e32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
; EG: 2143289344({{nan|1\.#QNAN0e\+00}})
define void @constant_fold_fmin_f32_nan_nan(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float 0x7FF8000000000000, float 0x7FF8000000000000) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_val_nan
; SI-NOT: v_min_f32_e32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32_val_nan(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float 1.0, float 0x7FF8000000000000) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_nan_val
; SI-NOT: v_min_f32_e32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32_nan_val(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float 0x7FF8000000000000, float 1.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_p0_p0
; SI-NOT: v_min_f32_e32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32_p0_p0(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float 0.0, float 0.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_p0_n0
; SI-NOT: v_min_f32_e32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32_p0_n0(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float 0.0, float -0.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_n0_p0
; SI-NOT: v_min_f32_e32
; SI: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32_n0_p0(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float -0.0, float 0.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @constant_fold_fmin_f32_n0_n0
; SI-NOT: v_min_f32_e32
; SI: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; SI: buffer_store_dword [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG-NOT: MIN_DX10
; EG: MOV {{.*}}[[OUT]], literal.{{[xy]}}
define void @constant_fold_fmin_f32_n0_n0(float addrspace(1)* %out) nounwind {
  %val = call float @llvm.minnum.f32(float -0.0, float -0.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @fmin_var_immediate_f32
; SI: v_min_f32_e64 {{v[0-9]+}}, 2.0, {{s[0-9]+}}

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MIN_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define void @fmin_var_immediate_f32(float addrspace(1)* %out, float %a) nounwind {
  %val = call float @llvm.minnum.f32(float %a, float 2.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @fmin_immediate_var_f32
; SI: v_min_f32_e64 {{v[0-9]+}}, 2.0, {{s[0-9]+}}

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MIN_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define void @fmin_immediate_var_f32(float addrspace(1)* %out, float %a) nounwind {
  %val = call float @llvm.minnum.f32(float 2.0, float %a) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @fmin_var_literal_f32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x42c60000
; SI: v_min_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MIN_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define void @fmin_var_literal_f32(float addrspace(1)* %out, float %a) nounwind {
  %val = call float @llvm.minnum.f32(float %a, float 99.0) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @fmin_literal_var_f32
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x42c60000
; SI: v_min_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, [[REG]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[OUT:T[0-9]+\.[XYZW]]]
; EG: MIN_DX10 {{.*}}[[OUT]], {{KC0\[[0-9]\].[XYZW]}}, literal.{{[xy]}}
define void @fmin_literal_var_f32(float addrspace(1)* %out, float %a) nounwind {
  %val = call float @llvm.minnum.f32(float 99.0, float %a) #0
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
