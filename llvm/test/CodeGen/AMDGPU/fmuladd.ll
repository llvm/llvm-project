; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

declare float @llvm.fmuladd.f32(float, float, float)
declare double @llvm.fmuladd.f64(double, double, double)
declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare float @llvm.fabs.f32(float) nounwind readnone

; CHECK-LABEL: {{^}}fmuladd_f32:
; CHECK: v_mac_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

define void @fmuladd_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                         float addrspace(1)* %in2, float addrspace(1)* %in3) {
   %r0 = load float, float addrspace(1)* %in1
   %r1 = load float, float addrspace(1)* %in2
   %r2 = load float, float addrspace(1)* %in3
   %r3 = tail call float @llvm.fmuladd.f32(float %r0, float %r1, float %r2)
   store float %r3, float addrspace(1)* %out
   ret void
}

; CHECK-LABEL: {{^}}fmuladd_f64:
; CHECK: v_fma_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}

define void @fmuladd_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                         double addrspace(1)* %in2, double addrspace(1)* %in3) {
   %r0 = load double, double addrspace(1)* %in1
   %r1 = load double, double addrspace(1)* %in2
   %r2 = load double, double addrspace(1)* %in3
   %r3 = tail call double @llvm.fmuladd.f64(double %r0, double %r1, double %r2)
   store double %r3, double addrspace(1)* %out
   ret void
}

; CHECK-LABEL: {{^}}fmuladd_2.0_a_b_f32
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fmuladd_2.0_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load float, float addrspace(1)* %gep.0
  %r2 = load float, float addrspace(1)* %gep.1

  %r3 = tail call float @llvm.fmuladd.f32(float 2.0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; CHECK-LABEL: {{^}}fmuladd_a_2.0_b_f32
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fmuladd_a_2.0_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load float, float addrspace(1)* %gep.0
  %r2 = load float, float addrspace(1)* %gep.1

  %r3 = tail call float @llvm.fmuladd.f32(float %r1, float 2.0, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

; CHECK-LABEL: {{^}}fadd_a_a_b_f32:
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fadd_a_a_b_f32(float addrspace(1)* %out,
                            float addrspace(1)* %in1,
                            float addrspace(1)* %in2) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r0 = load float, float addrspace(1)* %gep.0
  %r1 = load float, float addrspace(1)* %gep.1

  %add.0 = fadd float %r0, %r0
  %add.1 = fadd float %add.0, %r1
  store float %add.1, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}fadd_b_a_a_f32:
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fadd_b_a_a_f32(float addrspace(1)* %out,
                            float addrspace(1)* %in1,
                            float addrspace(1)* %in2) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r0 = load float, float addrspace(1)* %gep.0
  %r1 = load float, float addrspace(1)* %gep.1

  %add.0 = fadd float %r0, %r0
  %add.1 = fadd float %r1, %add.0
  store float %add.1, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}fmuladd_neg_2.0_a_b_f32
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], -2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fmuladd_neg_2.0_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load float, float addrspace(1)* %gep.0
  %r2 = load float, float addrspace(1)* %gep.1

  %r3 = tail call float @llvm.fmuladd.f32(float -2.0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}


; CHECK-LABEL: {{^}}fmuladd_neg_2.0_neg_a_b_f32
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], 2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fmuladd_neg_2.0_neg_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load float, float addrspace(1)* %gep.0
  %r2 = load float, float addrspace(1)* %gep.1

  %r1.fneg = fsub float -0.000000e+00, %r1

  %r3 = tail call float @llvm.fmuladd.f32(float -2.0, float %r1.fneg, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}


; CHECK-LABEL: {{^}}fmuladd_2.0_neg_a_b_f32
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mac_f32_e32 [[R2]], -2.0, [[R1]]
; CHECK: buffer_store_dword [[R2]]
define void @fmuladd_2.0_neg_a_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load float, float addrspace(1)* %gep.0
  %r2 = load float, float addrspace(1)* %gep.1

  %r1.fneg = fsub float -0.000000e+00, %r1

  %r3 = tail call float @llvm.fmuladd.f32(float 2.0, float %r1.fneg, float %r2)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}


; CHECK-LABEL: {{^}}fmuladd_2.0_a_neg_b_f32
; CHECK-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CHECK-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; CHECK: v_mad_f32 [[RESULT:v[0-9]+]], 2.0, [[R1]], -[[R2]]
; CHECK: buffer_store_dword [[RESULT]]
define void @fmuladd_2.0_a_neg_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load float, float addrspace(1)* %gep.0
  %r2 = load float, float addrspace(1)* %gep.1

  %r2.fneg = fsub float -0.000000e+00, %r2

  %r3 = tail call float @llvm.fmuladd.f32(float 2.0, float %r1, float %r2.fneg)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}
