; RUN: llc -mtriple=amdgpu9.00 < %s 2>&1 | FileCheck --implicit-check-not=warning -check-prefix=WARN %s
; RUN: llc -mtriple=amdgpu11.00 < %s 2>&1 | FileCheck --implicit-check-not=warning -check-prefix=GFX11 %s

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

; A 512-lane workgroup needs at least 2 waves per EU on gfx9 (8 wave64 waves
; over 4 SIMDs) and 4 on gfx11 (16 wave32 waves over 4 SIMDs). A smaller
; minimum is ignored in favor of the default range, so the requested maximum
; is not applied either.
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'ignored_target_max_1': desired occupancy was at most 1, final occupancy is 10
; GFX11: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'ignored_target_max_1': desired occupancy was at most 1, final occupancy is 16
define amdgpu_kernel void @ignored_target_max_1() #2 {
  ret void
}

; The same range is applied on gfx9 but ignored on gfx11.
; GFX11: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'ignored_target_max_2_gfx11': desired occupancy was at most 2, final occupancy is 16
define amdgpu_kernel void @ignored_target_max_2_gfx11() #3 {
  ret void
}

; No maximum is requested, and the default minimum satisfies the requested one.
define amdgpu_kernel void @no_target_max() #4 {
  ret void
}

; The requested range is applied, but on gfx11 VGPR usage cannot limit
; occupancy below 5 waves per EU, so the final occupancy exceeds the requested
; maximum without the request being ignored.
define amdgpu_kernel void @target_max_2_applied() #5 {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1024,1024" "amdgpu-waves-per-eu"="9" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="11" }
attributes #2 = { "amdgpu-flat-work-group-size"="1,512" "amdgpu-waves-per-eu"="1,1" }
attributes #3 = { "amdgpu-flat-work-group-size"="1,512" "amdgpu-waves-per-eu"="2,2" }
attributes #4 = { "amdgpu-flat-work-group-size"="1,512" "amdgpu-waves-per-eu"="1" }
attributes #5 = { "amdgpu-flat-work-group-size"="1,64" "amdgpu-waves-per-eu"="2,2" }
