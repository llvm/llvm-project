; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=WARN %s

; 1024 flat work group size across 2560 possible threads -> occupancy should be 8 max.
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'occupancy_8_target_9': desired occupancy was 9, final occupancy is 8
define amdgpu_kernel void @occupancy_8_target_9() #0 {
  ret void
}

; Impossible occupancy target
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'impossible_occupancy': desired occupancy was 11, final occupancy is 10
define amdgpu_kernel void @impossible_occupancy() #1 {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-waves-per-eu"="9" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="11" }
