; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

declare float @llvm.fabs.f32(float) #1
declare double @llvm.fabs.f64(double) #1

; GCN-LABEL: {{^}}test_isinf_pattern:
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x204{{$}}
; GCN: v_cmp_class_f32_e32 vcc, s{{[0-9]+}}, [[MASK]]
; GCN-NOT: v_cmp
; GCN: s_endpgm
define amdgpu_kernel void @test_isinf_pattern(ptr addrspace(1) nocapture %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x) #1
  %cmp = fcmp oeq float %fabs, 0x7FF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_not_isinf_pattern_0:
; GCN-NOT: v_cmp_class
; GCN: s_endpgm
define amdgpu_kernel void @test_not_isinf_pattern_0(ptr addrspace(1) nocapture %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x) #1
  %cmp = fcmp ueq float %fabs, 0x7FF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_not_isinf_pattern_1:
; GCN-NOT: v_cmp_class
; GCN: s_endpgm
define amdgpu_kernel void @test_not_isinf_pattern_1(ptr addrspace(1) nocapture %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x) #1
  %cmp = fcmp oeq float %fabs, 0xFFF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_isfinite_pattern_0:
; GCN-NOT: v_cmp
; GCN: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1f8{{$}}
; GCN: v_cmp_class_f32_e32 vcc, s{{[0-9]+}}, [[MASK]]
; GCN-NOT: v_cmp
; GCN: s_endpgm
define amdgpu_kernel void @test_isfinite_pattern_0(ptr addrspace(1) nocapture %out, float %x) #0 {
  %ord = fcmp ord float %x, 0.000000e+00
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #1
  %ninf = fcmp une float %x.fabs, 0x7FF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_isfinite_pattern_1:
; SI-NOT: v_cmp
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1f8{{$}}
; SI: v_cmp_class_f32_e32 vcc, s{{[0-9]+}}, [[MASK]]
; SI-NOT: v_cmp
; SI: s_endpgm
define amdgpu_kernel void @test_isfinite_pattern_1(ptr addrspace(1) nocapture %out, float %x) #0 {
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #3
  %cmpinf = fcmp one float %x.fabs, 0x7FF0000000000000
  %ext = zext i1 %cmpinf to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; Use negative infinity
; GCN-LABEL: {{^}}test_isfinite_not_pattern_0:
; GCN-NOT: v_cmp_class_f32
; GCN: s_endpgm
define amdgpu_kernel void @test_isfinite_not_pattern_0(ptr addrspace(1) nocapture %out, float %x) #0 {
  %ord = fcmp ord float %x, 0.000000e+00
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #1
  %ninf = fcmp une float %x.fabs, 0xFFF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; No fabs
; GCN-LABEL: {{^}}test_isfinite_not_pattern_1:
; GCN-NOT: v_cmp_class_f32
; GCN: s_endpgm
define amdgpu_kernel void @test_isfinite_not_pattern_1(ptr addrspace(1) nocapture %out, float %x) #0 {
  %ord = fcmp ord float %x, 0.000000e+00
  %ninf = fcmp une float %x, 0x7FF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; fabs of different value
; GCN-LABEL: {{^}}test_isfinite_not_pattern_2:
; GCN-NOT: v_cmp_class_f32
; GCN: s_endpgm
define amdgpu_kernel void @test_isfinite_not_pattern_2(ptr addrspace(1) nocapture %out, float %x, float %y) #0 {
  %ord = fcmp ord float %x, 0.000000e+00
  %x.fabs = tail call float @llvm.fabs.f32(float %y) #1
  %ninf = fcmp une float %x.fabs, 0x7FF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; Wrong ordered compare type
; GCN-LABEL: {{^}}test_isfinite_not_pattern_3:
; GCN-NOT: v_cmp_class_f32
; GCN: s_endpgm
define amdgpu_kernel void @test_isfinite_not_pattern_3(ptr addrspace(1) nocapture %out, float %x) #0 {
  %ord = fcmp uno float %x, 0.000000e+00
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #1
  %ninf = fcmp une float %x.fabs, 0x7FF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_isfinite_pattern_4:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x1f8
; GCN-DAG: v_cmp_class_f32_e32 vcc, [[X]], [[K]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @test_isfinite_pattern_4(ptr addrspace(1) nocapture %out, float %x) #0 {
  %ord = fcmp ord float %x, 0.000000e+00
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #1
  %ninf = fcmp one float %x.fabs, 0x7FF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_isfinite_pattern_4_commute_and:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x1f8
; GCN-DAG: v_cmp_class_f32_e32 vcc, [[X]], [[K]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @test_isfinite_pattern_4_commute_and(ptr addrspace(1) nocapture %out, float %x) #0 {
  %ord = fcmp ord float %x, 0.000000e+00
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #1
  %ninf = fcmp one float %x.fabs, 0x7FF0000000000000
  %and = and i1 %ninf, %ord
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_not_isfinite_pattern_4_wrong_ord_test:
; GCN-DAG: s_load_dword [[X:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[Y:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0x14|0x50}}

; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x1f8
; GCN-DAG: v_mov_b32_e32 [[VY:v[0-9]+]], [[Y]]

; SI-DAG: v_cmp_o_f32_e32 vcc, [[X]], [[VY]]
; SI-DAG: v_cmp_class_f32_e64 [[CLASS:s\[[0-9]+:[0-9]+\]]], [[X]], [[K]]
; SI: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[CLASS]]

; VI-DAG: v_cmp_o_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], [[X]], [[VY]]
; VI-DAG: v_cmp_class_f32_e32 vcc, [[X]], [[K]]
; VI: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], [[CMP]], vcc

; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[AND]]
define amdgpu_kernel void @test_not_isfinite_pattern_4_wrong_ord_test(ptr addrspace(1) nocapture %out, float %x, [8 x i32], float %y) #0 {
  %ord = fcmp ord float %x, %y
  %x.fabs = tail call float @llvm.fabs.f32(float %x) #1
  %ninf = fcmp one float %x.fabs, 0x7FF0000000000000
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_isinf_pattern_f16:
; SI-DAG: s_mov_b32 [[INF:s[0-9]+]], 0x7f800000
; SI-DAG: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], |s{{[0-9]+}}|
; SI: v_cmp_eq_f32_e32 vcc, [[INF]], [[CVT]]
; SI-NEXT: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc

; VI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x204{{$}}
; VI: v_cmp_class_f16_e32 vcc, s{{[0-9]+}}, [[MASK]]
; VI-NOT: v_cmp

; GCN: s_endpgm
define amdgpu_kernel void @test_isinf_pattern_f16(ptr addrspace(1) nocapture %out, half %x) #0 {
  %fabs = tail call half @llvm.fabs.f16(half %x) #1
  %cmp = fcmp oeq half %fabs, 0xH7C00
  %ext = zext i1 %cmp to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_isfinite_pattern_0_f16:
; SI-DAG: s_movk_i32 [[MASK:s[0-9]+]], 0x1f8
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_class_f32_e64 [[CLASS:s\[[0-9]+:[0-9]+\]]], [[CVT]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[CLASS]]

; VI-NOT: v_cmp
; VI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1f8{{$}}
; VI: v_cmp_class_f16_e32 vcc, s{{[0-9]+}}, [[MASK]]
; VI-NOT: v_cmp

; GCN: s_endpgm
define amdgpu_kernel void @test_isfinite_pattern_0_f16(ptr addrspace(1) nocapture %out, half %x) #0 {
  %ord = fcmp ord half %x, 0.0
  %x.fabs = tail call half @llvm.fabs.f16(half %x) #1
  %ninf = fcmp une half %x.fabs, 0xH7C00
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_isfinite_pattern_4_f16:
; SI-DAG: s_movk_i32 [[MASK:s[0-9]+]], 0x1f8
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_class_f32_e64 [[CLASS:s\[[0-9]+:[0-9]+\]]], [[CVT]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[CLASS]]

; VI-DAG: s_load_dword [[X:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1f8
; VI: v_cmp_class_f16_e32 vcc, [[X]], [[MASK]]
; VI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @test_isfinite_pattern_4_f16(ptr addrspace(1) nocapture %out, half %x) #0 {
  %ord = fcmp ord half %x, 0.0
  %x.fabs = tail call half @llvm.fabs.f16(half %x) #1
  %ninf = fcmp one half %x.fabs, 0xH7C00
  %and = and i1 %ord, %ninf
  %ext = zext i1 %and to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}

declare half @llvm.fabs.f16(half) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
