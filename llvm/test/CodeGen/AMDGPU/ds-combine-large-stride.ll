; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: ds_read32_combine_stride_400:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B2:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B3:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]

; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x200, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B2:v[0-9]+]], 0x400, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B3:v[0-9]+]], 0x800, [[BASE]]

; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[BASE]] offset1:100
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:72 offset1:172
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B2]] offset0:144 offset1:244
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B3]] offset0:88 offset1:188
define amdgpu_kernel void @ds_read32_combine_stride_400(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = load float, ptr addrspace(3) %arg, align 4
  %tmp2 = fadd float %tmp, 0.000000e+00
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 100
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp5 = fadd float %tmp2, %tmp4
  %tmp6 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 200
  %tmp7 = load float, ptr addrspace(3) %tmp6, align 4
  %tmp8 = fadd float %tmp5, %tmp7
  %tmp9 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 300
  %tmp10 = load float, ptr addrspace(3) %tmp9, align 4
  %tmp11 = fadd float %tmp8, %tmp10
  %tmp12 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 400
  %tmp13 = load float, ptr addrspace(3) %tmp12, align 4
  %tmp14 = fadd float %tmp11, %tmp13
  %tmp15 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 500
  %tmp16 = load float, ptr addrspace(3) %tmp15, align 4
  %tmp17 = fadd float %tmp14, %tmp16
  %tmp18 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 600
  %tmp19 = load float, ptr addrspace(3) %tmp18, align 4
  %tmp20 = fadd float %tmp17, %tmp19
  %tmp21 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 700
  %tmp22 = load float, ptr addrspace(3) %tmp21, align 4
  %tmp23 = fadd float %tmp20, %tmp22
  store float %tmp23, ptr %arg1, align 4
  ret void
}

; GCN-LABEL: ds_read32_combine_stride_20:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B2:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]

; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x400, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B2:v[0-9]+]], 0x800, [[BASE]]

; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:144 offset1:164
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:184 offset1:204
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:224 offset1:244
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B2]] offset0:8 offset1:28
define amdgpu_kernel void @ds_read32_combine_stride_20(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 400
  %tmp1 = load float, ptr addrspace(3) %tmp, align 4
  %tmp2 = fadd float %tmp1, 0.000000e+00
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 420
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp5 = fadd float %tmp2, %tmp4
  %tmp6 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 440
  %tmp7 = load float, ptr addrspace(3) %tmp6, align 4
  %tmp8 = fadd float %tmp5, %tmp7
  %tmp9 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 460
  %tmp10 = load float, ptr addrspace(3) %tmp9, align 4
  %tmp11 = fadd float %tmp8, %tmp10
  %tmp12 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 480
  %tmp13 = load float, ptr addrspace(3) %tmp12, align 4
  %tmp14 = fadd float %tmp11, %tmp13
  %tmp15 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 500
  %tmp16 = load float, ptr addrspace(3) %tmp15, align 4
  %tmp17 = fadd float %tmp14, %tmp16
  %tmp18 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 520
  %tmp19 = load float, ptr addrspace(3) %tmp18, align 4
  %tmp20 = fadd float %tmp17, %tmp19
  %tmp21 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 540
  %tmp22 = load float, ptr addrspace(3) %tmp21, align 4
  %tmp23 = fadd float %tmp20, %tmp22
  store float %tmp23, ptr %arg1, align 4
  ret void
}

; GCN-LABEL: ds_read32_combine_stride_400_back:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B2:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B3:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]

; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x800, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B2:v[0-9]+]], 0x400, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B3:v[0-9]+]], 0x200, [[BASE]]

; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[BASE]] offset1:100
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:88 offset1:188
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B2]] offset0:144 offset1:244
; GCN-DAG: ds_read2_b32  v[{{[0-9]+:[0-9]+}}], [[B3]] offset0:72 offset1:172
define amdgpu_kernel void @ds_read32_combine_stride_400_back(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 700
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = fadd float %tmp2, 0.000000e+00
  %tmp4 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 600
  %tmp5 = load float, ptr addrspace(3) %tmp4, align 4
  %tmp6 = fadd float %tmp3, %tmp5
  %tmp7 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 500
  %tmp8 = load float, ptr addrspace(3) %tmp7, align 4
  %tmp9 = fadd float %tmp6, %tmp8
  %tmp10 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 400
  %tmp11 = load float, ptr addrspace(3) %tmp10, align 4
  %tmp12 = fadd float %tmp9, %tmp11
  %tmp13 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 300
  %tmp14 = load float, ptr addrspace(3) %tmp13, align 4
  %tmp15 = fadd float %tmp12, %tmp14
  %tmp16 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 200
  %tmp17 = load float, ptr addrspace(3) %tmp16, align 4
  %tmp18 = fadd float %tmp15, %tmp17
  %tmp19 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 100
  %tmp20 = load float, ptr addrspace(3) %tmp19, align 4
  %tmp21 = fadd float %tmp18, %tmp20
  %tmp22 = load float, ptr addrspace(3) %arg, align 4
  %tmp23 = fadd float %tmp21, %tmp22
  store float %tmp23, ptr %arg1, align 4
  ret void
}

; GCN-LABEL: ds_read32_combine_stride_8192:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]
; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[BASE]] offset1:32
; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[BASE]] offset0:64 offset1:96
; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[BASE]] offset0:128 offset1:160
; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[BASE]] offset0:192 offset1:224
define amdgpu_kernel void @ds_read32_combine_stride_8192(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = load float, ptr addrspace(3) %arg, align 4
  %tmp2 = fadd float %tmp, 0.000000e+00
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2048
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp5 = fadd float %tmp2, %tmp4
  %tmp6 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 4096
  %tmp7 = load float, ptr addrspace(3) %tmp6, align 4
  %tmp8 = fadd float %tmp5, %tmp7
  %tmp9 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 6144
  %tmp10 = load float, ptr addrspace(3) %tmp9, align 4
  %tmp11 = fadd float %tmp8, %tmp10
  %tmp12 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 8192
  %tmp13 = load float, ptr addrspace(3) %tmp12, align 4
  %tmp14 = fadd float %tmp11, %tmp13
  %tmp15 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 10240
  %tmp16 = load float, ptr addrspace(3) %tmp15, align 4
  %tmp17 = fadd float %tmp14, %tmp16
  %tmp18 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 12288
  %tmp19 = load float, ptr addrspace(3) %tmp18, align 4
  %tmp20 = fadd float %tmp17, %tmp19
  %tmp21 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 14336
  %tmp22 = load float, ptr addrspace(3) %tmp21, align 4
  %tmp23 = fadd float %tmp20, %tmp22
  store float %tmp23, ptr %arg1, align 4
  ret void
}

; GCN-LABEL: ds_read32_combine_stride_8192_shifted:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, 8, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 8, [[BASE]]

; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[B1]] offset1:32
; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:64 offset1:96
; GCN-DAG: ds_read2st64_b32 v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:128 offset1:160
define amdgpu_kernel void @ds_read32_combine_stride_8192_shifted(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = fadd float %tmp2, 0.000000e+00
  %tmp4 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2050
  %tmp5 = load float, ptr addrspace(3) %tmp4, align 4
  %tmp6 = fadd float %tmp3, %tmp5
  %tmp7 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 4098
  %tmp8 = load float, ptr addrspace(3) %tmp7, align 4
  %tmp9 = fadd float %tmp6, %tmp8
  %tmp10 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 6146
  %tmp11 = load float, ptr addrspace(3) %tmp10, align 4
  %tmp12 = fadd float %tmp9, %tmp11
  %tmp13 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 8194
  %tmp14 = load float, ptr addrspace(3) %tmp13, align 4
  %tmp15 = fadd float %tmp12, %tmp14
  %tmp16 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 10242
  %tmp17 = load float, ptr addrspace(3) %tmp16, align 4
  %tmp18 = fadd float %tmp15, %tmp17
  store float %tmp18, ptr %arg1, align 4
  ret void
}

; GCN-LABEL: ds_read64_combine_stride_400:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x800, [[BASE]]

; GCN-DAG: ds_read2_b64  v[{{[0-9]+:[0-9]+}}], [[BASE]] offset1:50
; GCN-DAG: ds_read2_b64  v[{{[0-9]+:[0-9]+}}], [[BASE]] offset0:100 offset1:150
; GCN-DAG: ds_read2_b64  v[{{[0-9]+:[0-9]+}}], [[BASE]] offset0:200 offset1:250
; GCN-DAG: ds_read2_b64  v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:44 offset1:94
define amdgpu_kernel void @ds_read64_combine_stride_400(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = load double, ptr addrspace(3) %arg, align 8
  %tmp2 = fadd double %tmp, 0.000000e+00
  %tmp3 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 50
  %tmp4 = load double, ptr addrspace(3) %tmp3, align 8
  %tmp5 = fadd double %tmp2, %tmp4
  %tmp6 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 100
  %tmp7 = load double, ptr addrspace(3) %tmp6, align 8
  %tmp8 = fadd double %tmp5, %tmp7
  %tmp9 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 150
  %tmp10 = load double, ptr addrspace(3) %tmp9, align 8
  %tmp11 = fadd double %tmp8, %tmp10
  %tmp12 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 200
  %tmp13 = load double, ptr addrspace(3) %tmp12, align 8
  %tmp14 = fadd double %tmp11, %tmp13
  %tmp15 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 250
  %tmp16 = load double, ptr addrspace(3) %tmp15, align 8
  %tmp17 = fadd double %tmp14, %tmp16
  %tmp18 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 300
  %tmp19 = load double, ptr addrspace(3) %tmp18, align 8
  %tmp20 = fadd double %tmp17, %tmp19
  %tmp21 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 350
  %tmp22 = load double, ptr addrspace(3) %tmp21, align 8
  %tmp23 = fadd double %tmp20, %tmp22
  store double %tmp23, ptr %arg1, align 8
  ret void
}

; GCN-LABEL: ds_read64_combine_stride_8192_shifted:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, 8, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 8, [[BASE]]

; GCN-DAG: ds_read2st64_b64 v[{{[0-9]+:[0-9]+}}], [[B1]] offset1:16
; GCN-DAG: ds_read2st64_b64 v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:32 offset1:48
; GCN-DAG: ds_read2st64_b64 v[{{[0-9]+:[0-9]+}}], [[B1]] offset0:64 offset1:80
define amdgpu_kernel void @ds_read64_combine_stride_8192_shifted(ptr addrspace(3) nocapture readonly %arg, ptr nocapture %arg1) {
bb:
  %tmp = getelementptr inbounds double, ptr addrspace(3) %arg, i32 1
  %tmp2 = load double, ptr addrspace(3) %tmp, align 8
  %tmp3 = fadd double %tmp2, 0.000000e+00
  %tmp4 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 1025
  %tmp5 = load double, ptr addrspace(3) %tmp4, align 8
  %tmp6 = fadd double %tmp3, %tmp5
  %tmp7 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 2049
  %tmp8 = load double, ptr addrspace(3) %tmp7, align 8
  %tmp9 = fadd double %tmp6, %tmp8
  %tmp10 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 3073
  %tmp11 = load double, ptr addrspace(3) %tmp10, align 8
  %tmp12 = fadd double %tmp9, %tmp11
  %tmp13 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 4097
  %tmp14 = load double, ptr addrspace(3) %tmp13, align 8
  %tmp15 = fadd double %tmp12, %tmp14
  %tmp16 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 5121
  %tmp17 = load double, ptr addrspace(3) %tmp16, align 8
  %tmp18 = fadd double %tmp15, %tmp17
  store double %tmp18, ptr %arg1, align 8
  ret void
}

; GCN-LABEL: ds_write32_combine_stride_400:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B2:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B3:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]

; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x200, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B2:v[0-9]+]], 0x400, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B3:v[0-9]+]], 0x800, [[BASE]]

; GCN-DAG: ds_write2_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset1:100
; GCN-DAG: ds_write2_b32 [[B1]], v{{[0-9]+}}, v{{[0-9]+}} offset0:72 offset1:172
; GCN-DAG: ds_write2_b32 [[B2]], v{{[0-9]+}}, v{{[0-9]+}} offset0:144 offset1:244
; GCN-DAG: ds_write2_b32 [[B3]], v{{[0-9]+}}, v{{[0-9]+}} offset0:88 offset1:188
define amdgpu_kernel void @ds_write32_combine_stride_400(ptr addrspace(3) nocapture %arg) {
bb:
  store float 1.000000e+00, ptr addrspace(3) %arg, align 4
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 100
  store float 1.000000e+00, ptr addrspace(3) %tmp, align 4
  %tmp1 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 200
  store float 1.000000e+00, ptr addrspace(3) %tmp1, align 4
  %tmp2 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 300
  store float 1.000000e+00, ptr addrspace(3) %tmp2, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 400
  store float 1.000000e+00, ptr addrspace(3) %tmp3, align 4
  %tmp4 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 500
  store float 1.000000e+00, ptr addrspace(3) %tmp4, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 600
  store float 1.000000e+00, ptr addrspace(3) %tmp5, align 4
  %tmp6 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 700
  store float 1.000000e+00, ptr addrspace(3) %tmp6, align 4
  ret void
}

; GCN-LABEL: ds_write32_combine_stride_400_back:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B2:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; VI-DAG: v_add_u32_e32 [[B3:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]

; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x800, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B2:v[0-9]+]], 0x400, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B3:v[0-9]+]], 0x200, [[BASE]]

; GCN-DAG: ds_write2_b32 [[B1]], v{{[0-9]+}}, v{{[0-9]+}} offset0:88 offset1:188
; GCN-DAG: ds_write2_b32 [[B2]], v{{[0-9]+}}, v{{[0-9]+}} offset0:144 offset1:244
; GCN-DAG: ds_write2_b32 [[B3]], v{{[0-9]+}}, v{{[0-9]+}} offset0:72 offset1:172
; GCN-DAG: ds_write2_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset1:100
define amdgpu_kernel void @ds_write32_combine_stride_400_back(ptr addrspace(3) nocapture %arg) {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 700
  store float 1.000000e+00, ptr addrspace(3) %tmp, align 4
  %tmp1 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 600
  store float 1.000000e+00, ptr addrspace(3) %tmp1, align 4
  %tmp2 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 500
  store float 1.000000e+00, ptr addrspace(3) %tmp2, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 400
  store float 1.000000e+00, ptr addrspace(3) %tmp3, align 4
  %tmp4 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 300
  store float 1.000000e+00, ptr addrspace(3) %tmp4, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 200
  store float 1.000000e+00, ptr addrspace(3) %tmp5, align 4
  %tmp6 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 100
  store float 1.000000e+00, ptr addrspace(3) %tmp6, align 4
  store float 1.000000e+00, ptr addrspace(3) %arg, align 4
  ret void
}

; GCN-LABEL: ds_write32_combine_stride_8192:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]
; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset1:32
; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset0:64 offset1:96
; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset0:128 offset1:160
; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset0:192 offset1:224
define amdgpu_kernel void @ds_write32_combine_stride_8192(ptr addrspace(3) nocapture %arg) {
bb:
  store float 1.000000e+00, ptr addrspace(3) %arg, align 4
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2048
  store float 1.000000e+00, ptr addrspace(3) %tmp, align 4
  %tmp1 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 4096
  store float 1.000000e+00, ptr addrspace(3) %tmp1, align 4
  %tmp2 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 6144
  store float 1.000000e+00, ptr addrspace(3) %tmp2, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 8192
  store float 1.000000e+00, ptr addrspace(3) %tmp3, align 4
  %tmp4 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 10240
  store float 1.000000e+00, ptr addrspace(3) %tmp4, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 12288
  store float 1.000000e+00, ptr addrspace(3) %tmp5, align 4
  %tmp6 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 14336
  store float 1.000000e+00, ptr addrspace(3) %tmp6, align 4
  ret void
}

; GCN-LABEL: ds_write32_combine_stride_8192_shifted:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[BASE:v[0-9]+]], vcc, 4, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[BASE:v[0-9]+]], 4, [[BASE]]

; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset1:32
; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset0:64 offset1:96
; GCN-DAG: ds_write2st64_b32 [[BASE]], v{{[0-9]+}}, v{{[0-9]+}} offset0:128 offset1:160
define amdgpu_kernel void @ds_write32_combine_stride_8192_shifted(ptr addrspace(3) nocapture %arg) {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 1
  store float 1.000000e+00, ptr addrspace(3) %tmp, align 4
  %tmp1 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2049
  store float 1.000000e+00, ptr addrspace(3) %tmp1, align 4
  %tmp2 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 4097
  store float 1.000000e+00, ptr addrspace(3) %tmp2, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 6145
  store float 1.000000e+00, ptr addrspace(3) %tmp3, align 4
  %tmp4 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 8193
  store float 1.000000e+00, ptr addrspace(3) %tmp4, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 10241
  store float 1.000000e+00, ptr addrspace(3) %tmp5, align 4
  ret void
}

; GCN-LABEL: ds_write64_combine_stride_400:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[B1:v[0-9]+]], vcc, {{s[0-9]+}}, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[B1:v[0-9]+]], 0x800, [[BASE]]

; GCN-DAG: ds_write2_b64 [[BASE]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset1:50
; GCN-DAG: ds_write2_b64 [[BASE]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset0:100 offset1:150
; GCN-DAG: ds_write2_b64 [[BASE]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset0:200 offset1:250
; GCN-DAG: ds_write2_b64 [[B1]],   v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset0:44 offset1:94
define amdgpu_kernel void @ds_write64_combine_stride_400(ptr addrspace(3) nocapture %arg) {
bb:
  store double 1.000000e+00, ptr addrspace(3) %arg, align 8
  %tmp = getelementptr inbounds double, ptr addrspace(3) %arg, i32 50
  store double 1.000000e+00, ptr addrspace(3) %tmp, align 8
  %tmp1 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 100
  store double 1.000000e+00, ptr addrspace(3) %tmp1, align 8
  %tmp2 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 150
  store double 1.000000e+00, ptr addrspace(3) %tmp2, align 8
  %tmp3 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 200
  store double 1.000000e+00, ptr addrspace(3) %tmp3, align 8
  %tmp4 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 250
  store double 1.000000e+00, ptr addrspace(3) %tmp4, align 8
  %tmp5 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 300
  store double 1.000000e+00, ptr addrspace(3) %tmp5, align 8
  %tmp6 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 350
  store double 1.000000e+00, ptr addrspace(3) %tmp6, align 8
  ret void
}

; GCN-LABEL: ds_write64_combine_stride_8192_shifted:
; GCN:     s_load_dword [[ARG:s[0-9]+]], s[4:5], 0x0
; GCN:     v_mov_b32_e32 [[BASE:v[0-9]+]], [[ARG]]

; VI-DAG: v_add_u32_e32 [[BASE]], vcc, 8, [[BASE]]
; GFX9-DAG: v_add_u32_e32 [[BASE]], 8, [[BASE]]

; GCN-DAG: ds_write2st64_b64 [[BASE]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset1:16
; GCN-DAG: ds_write2st64_b64 [[BASE]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset0:32 offset1:48
; GCN-DAG: ds_write2st64_b64 [[BASE]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset0:64 offset1:80
define amdgpu_kernel void @ds_write64_combine_stride_8192_shifted(ptr addrspace(3) nocapture %arg) {
bb:
  %tmp = getelementptr inbounds double, ptr addrspace(3) %arg, i32 1
  store double 1.000000e+00, ptr addrspace(3) %tmp, align 8
  %tmp1 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 1025
  store double 1.000000e+00, ptr addrspace(3) %tmp1, align 8
  %tmp2 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 2049
  store double 1.000000e+00, ptr addrspace(3) %tmp2, align 8
  %tmp3 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 3073
  store double 1.000000e+00, ptr addrspace(3) %tmp3, align 8
  %tmp4 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 4097
  store double 1.000000e+00, ptr addrspace(3) %tmp4, align 8
  %tmp5 = getelementptr inbounds double, ptr addrspace(3) %arg, i32 5121
  store double 1.000000e+00, ptr addrspace(3) %tmp5, align 8
  ret void
}
