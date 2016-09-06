; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_default_si:
; GCN: FloatMode: 192
; GCN: IeeeMode: 0
define void @test_default_si(float addrspace(1)* %out0, double addrspace(1)* %out1) #0 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_default_vi:
; GCN: FloatMode: 192
; GCN: IeeeMode: 0
define void @test_default_vi(float addrspace(1)* %out0, double addrspace(1)* %out1) #1 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f64_denormals:
; GCN: FloatMode: 192
; GCN: IeeeMode: 0
define void @test_f64_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #2 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_denormals:
; GCNL: FloatMode: 48
; GCN: IeeeMode: 0
define void @test_f32_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #3 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_f32_f64_denormals:
; GCN: FloatMode: 240
; GCN: IeeeMode: 0
define void @test_f32_f64_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #4 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals
; GCN: FloatMode: 0
; GCN: IeeeMode: 0
define void @test_no_denormals(float addrspace(1)* %out0, double addrspace(1)* %out1) #5 {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}

attributes #0 = { nounwind "target-cpu"="tahiti" }
attributes #1 = { nounwind "target-cpu"="fiji" }
attributes #2 = { nounwind "target-features"="+fp64-denormals" }
attributes #3 = { nounwind "target-features"="+fp32-denormals" }
attributes #4 = { nounwind "target-features"="+fp32-denormals,+fp64-denormals" }
attributes #5 = { nounwind "target-features"="-fp32-denormals,-fp64-denormals" }
