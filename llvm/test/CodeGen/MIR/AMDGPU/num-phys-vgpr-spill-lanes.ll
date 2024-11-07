; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa --stop-after=prologepilog -o - %s | FileCheck %s

; Spill the PC SGPR30_31 and FP to physical VGPR

define void @test() #0 {
; CHECK: machineFunctionInfo
; CHECK: numPhysicalVGPRSpillLanes: 3
entry:
  %call = call i32 @ext_func()
  ret void
}

declare i32 @ext_func();

attributes #0 = { nounwind "amdgpu-num-vgpr"="41" "amdgpu-num-sgpr"="34" }
