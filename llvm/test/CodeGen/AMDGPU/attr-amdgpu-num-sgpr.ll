; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}max_18_sgprs:
; CHECK: SGPRBlocks: 1
; CHECK: NumSGPRsForWavesPerEU: 13
define void @max_18_sgprs(i32 addrspace(1)* %out1,
                          i32 addrspace(1)* %out2,
                          i32 addrspace(1)* %out3,
                          i32 addrspace(1)* %out4,
                          i32 %one, i32 %two, i32 %three, i32 %four) #0 {
  store i32 %one, i32 addrspace(1)* %out1
  store i32 %two, i32 addrspace(1)* %out2
  store i32 %three, i32 addrspace(1)* %out3
  store i32 %four, i32 addrspace(1)* %out4
  ret void
}
attributes #0 = {"amdgpu-num-sgpr"="18"}
