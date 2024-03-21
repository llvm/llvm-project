; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_default_si:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_default_si(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "target-cpu"="tahiti" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_default_vi:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_default_vi(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "target-cpu"="fiji" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_denormals:
; GCNL: FloatMode: 48
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_f32_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math-f32"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_f32_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals
; GCN: FloatMode: 0
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_no_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="preserve-sign,preserve-sign" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f16_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_f16_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="ieee,ieee" {
  store half 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_f16_f64_denormals:
; GCN: FloatMode: 48
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_no_f16_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math-f32"="ieee,ieee" "denormal-fp-math"="preserve-sign,preserve-sign" {
  store half 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f16_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_f32_f16_f64_denormals(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %out2) nounwind "denormal-fp-math"="ieee,ieee" {
  store half 0.0, ptr addrspace(1) %out0
  store float 0.0, ptr addrspace(1) %out1
  store double 0.0, ptr addrspace(1) %out2
  ret void
}

; GCN-LABEL: {{^}}test_just_f32_attr_flush
; GCN: FloatMode: 192
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_just_f32_attr_flush(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_flush_all_outputs:
; GCN: FloatMode: 80
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_flush_all_outputs(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="preserve-sign,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_flush_all_inputs:
; GCN: FloatMode: 160
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_flush_all_inputs(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="ieee,preserve-sign" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_flush_f32_inputs:
; GCN: FloatMode: 224
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_flush_f32_inputs(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math-f32"="ieee,preserve-sign" "denormal-fp-math"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_flush_f32_outputs:
; GCN: FloatMode: 208
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_flush_f32_outputs(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math-f32"="preserve-sign,ieee" "denormal-fp-math"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_flush_f64_inputs:
; GCN: FloatMode: 176
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_flush_f64_inputs(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="ieee,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}test_flush_f64_outputs:
; GCN: FloatMode: 112
; GCN: IeeeMode: 1
define amdgpu_kernel void @test_flush_f64_outputs(ptr addrspace(1) %out0, ptr addrspace(1) %out1) nounwind "denormal-fp-math"="preserve-sign,ieee" "denormal-fp-math-f32"="ieee,ieee" {
  store float 0.0, ptr addrspace(1) %out0
  store double 0.0, ptr addrspace(1) %out1
  ret void
}

; GCN-LABEL: {{^}}kill_gs_const:
; GCN: FloatMode: 240
; GCN: IeeeMode: 0
define amdgpu_gs void @kill_gs_const() {
main_body:
  %cmp0 = icmp ule i32 0, 3
  call void @llvm.amdgcn.kill(i1 %cmp0)
  %cmp1 = icmp ule i32 3, 0
  call void @llvm.amdgcn.kill(i1 %cmp1)
  ret void
}

; GCN-LABEL: {{^}}kill_vcc_implicit_def:
; GCN: FloatMode: 240
; GCN: IeeeMode: 0
define amdgpu_ps float @kill_vcc_implicit_def(ptr addrspace(4) inreg, ptr addrspace(4) inreg, ptr addrspace(4) inreg, ptr addrspace(4) inreg, float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, i32, float, float) {
entry:
  %tmp0 = fcmp olt float %13, 0.0
  call void @llvm.amdgcn.kill(i1 %tmp0)
  %tmp1 = select i1 %tmp0, float 1.0, float 0.0
  ret float %tmp1
}

declare void @llvm.amdgcn.kill(i1)
