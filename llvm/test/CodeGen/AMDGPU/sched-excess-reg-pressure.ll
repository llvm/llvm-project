; RUN: llc -mtriple=amdgpu12.50 -stop-after=machine-scheduler -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.50 -stop-after=machine-scheduler -o - %s | FileCheck %s

; The scheduler records in the machine function info whether some region has
; register pressure above the allocatable limits. The register allocator uses
; this to disable local reassignment for such functions.

; CHECK-LABEL: name: low_pressure
; CHECK: machineFunctionInfo:
; CHECK: hasExcessRegPressure: false

; CHECK-LABEL: name: excess_pressure
; CHECK: machineFunctionInfo:
; CHECK: hasExcessRegPressure: true

define amdgpu_kernel void @low_pressure(ptr addrspace(1) %p) {
  %a = load volatile <4 x float>, ptr addrspace(1) %p
  %b = load volatile <4 x float>, ptr addrspace(1) %p
  %s = fadd <4 x float> %a, %b
  store <4 x float> %s, ptr addrspace(1) %p
  ret void
}

define amdgpu_kernel void @excess_pressure(ptr addrspace(1) %p) #0 {
  %a = load volatile <32 x float>, ptr addrspace(1) %p
  %b = load volatile <32 x float>, ptr addrspace(1) %p
  %s = fadd <32 x float> %a, %b
  store <32 x float> %s, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-num-vgpr"="32" }
