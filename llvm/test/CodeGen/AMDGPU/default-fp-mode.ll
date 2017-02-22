; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_default_si:
; GCN: FloatMode: 192
; GCN: IeeeMode: 1
define void @test_default_si(float addrspace(1)* %out0, double addrspace(1)* %out1) #0 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_default_vi:
; GCN: FloatMode: 192
; GCN: IeeeMode: 1
define void @test_default_vi(float addrspace(1)* %out0, double addrspace(1)* %out1) #1 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f64_denormals:
; GCN: FloatMode: 192
; GCN: IeeeMode: 1
define void @test_f64_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #2 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_denormals:
; GCNL: FloatMode: 48
; GCN: IeeeMode: 1
define void @test_f32_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #3 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define void @test_f32_f64_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #4 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals
; GCN: FloatMode: 0
; GCN: IeeeMode: 1
define void @test_no_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #5 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f16_f64_denormals:
; GCN: FloatMode: 192
; GCN: IeeeMode: 1
define void @test_f16_f64_denormals(half addrspace(1)* %out0, double addrspace(1)* %out1) #6 {
  store half 0.0, half addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_f16_f64_denormals:
; GCN: FloatMode: 0
; GCN: IeeeMode: 1
define void @test_no_f16_f64_denormals(half addrspace(1)* %out0, double addrspace(1)* %out1) #7 {
  store half 0.0, half addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f16_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 1
define void @test_f32_f16_f64_denormals(half addrspace(1)* %out0, float addrspace(1)* %out1, double addrspace(1)* %out2) #8 {
  store half 0.0, half addrspace(1)* %out0
  store float 0.0, float addrspace(1)* %out1
  store double 0.0, double addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}kill_gs_const:
; GCN: IeeeMode: 0
define amdgpu_gs void @kill_gs_const() {
main_body:
  %0 = icmp ule i32 0, 3
  %1 = select i1 %0, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %1)
  %2 = icmp ule i32 3, 0
  %3 = select i1 %2, float 1.000000e+00, float -1.000000e+00
  call void @llvm.AMDGPU.kill(float %3)
  ret void
}

; GCN-LABEL: {{^}}kill_vcc_implicit_def:
; GCN: IeeeMode: 0
define amdgpu_ps float @kill_vcc_implicit_def([6 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <4 x i32>] addrspace(2)* byval, [34 x <8 x i32>] addrspace(2)* byval, float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, i32, float, float) {
entry:
  %tmp0 = fcmp olt float %13, 0.0
  call void @llvm.AMDGPU.kill(float %14)
  %tmp1 = select i1 %tmp0, float 1.0, float 0.0
  ret float %tmp1
}

declare void @llvm.AMDGPU.kill(float)

attributes #0 = { nounwind "target-cpu"="tahiti" }
attributes #1 = { nounwind "target-cpu"="fiji" }
attributes #2 = { nounwind "target-features"="+fp64-denormals" }
attributes #3 = { nounwind "target-features"="+fp32-denormals" }
attributes #4 = { nounwind "target-features"="+fp32-denormals,+fp64-denormals" }
attributes #5 = { nounwind "target-features"="-fp32-denormals,-fp64-fp16-denormals" }
attributes #6 = { nounwind "target-features"="+fp64-fp16-denormals" }
attributes #7 = { nounwind "target-features"="-fp64-fp16-denormals" }
attributes #8 = { nounwind "target-features"="+fp32-denormals,+fp64-fp16-denormals" }
