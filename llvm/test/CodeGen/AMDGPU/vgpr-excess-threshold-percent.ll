; REQUIRES: asserts

; gfx942 tests with different threshold percentages
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -amdgpu-vgpr-threshold-percent=40  -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX942-40  %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -amdgpu-vgpr-threshold-percent=60  -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX942-60  %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -amdgpu-vgpr-threshold-percent=80  -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX942-80  %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -amdgpu-vgpr-threshold-percent=100 -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX942-100 %s

; gfx1250 tests with coexec scheduling strategy and different threshold percentages
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -mattr=+real-true16 --amdgpu-sched-strategy=coexec -amdgpu-vgpr-threshold-percent=40  -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX1250-40  %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -mattr=+real-true16 --amdgpu-sched-strategy=coexec -amdgpu-vgpr-threshold-percent=60  -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX1250-60  %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -mattr=+real-true16 --amdgpu-sched-strategy=coexec -amdgpu-vgpr-threshold-percent=80  -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX1250-80  %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -mattr=+real-true16 --amdgpu-sched-strategy=coexec -amdgpu-vgpr-threshold-percent=100 -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefixes=GFX1250-100 %s

; Test that -amdgpu-vgpr-threshold-percent affects VGPRExcessLimit and VGPRCriticalLimit
; for functions with different waves-per-eu targets.

; waves-per-eu=4,4
; GFX942-40:  test_waves_4
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 64 -> 26. VGPRCriticalLimit: 64 -> 26
; GFX942-60:  test_waves_4
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 64 -> 39. VGPRCriticalLimit: 64 -> 39
; GFX942-80:  test_waves_4
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 64 -> 52. VGPRCriticalLimit: 64 -> 52
; GFX942-100: test_waves_4
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 64 -> 64. VGPRCriticalLimit: 64 -> 64

; GFX1250-40:  test_waves_4
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 256 -> 103. VGPRCriticalLimit: 256 -> 103
; GFX1250-60:  test_waves_4
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 256 -> 154. VGPRCriticalLimit: 256 -> 154
; GFX1250-80:  test_waves_4
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 256 -> 205. VGPRCriticalLimit: 256 -> 205
; GFX1250-100: test_waves_4
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 256 -> 256. VGPRCriticalLimit: 256 -> 256
define amdgpu_kernel void @test_waves_4(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

; waves-per-eu=5,5
; GFX942-40:  test_waves_5
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 48 -> 20. VGPRCriticalLimit: 48 -> 20
; GFX942-60:  test_waves_5
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 48 -> 29. VGPRCriticalLimit: 48 -> 29
; GFX942-80:  test_waves_5
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 48 -> 39. VGPRCriticalLimit: 48 -> 39
; GFX942-100: test_waves_5
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 48 -> 48. VGPRCriticalLimit: 48 -> 48

; GFX1250-40:  test_waves_5
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 192 -> 77. VGPRCriticalLimit: 192 -> 77
; GFX1250-60:  test_waves_5
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 192 -> 116. VGPRCriticalLimit: 192 -> 116
; GFX1250-80:  test_waves_5
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 192 -> 154. VGPRCriticalLimit: 192 -> 154
; GFX1250-100: test_waves_5
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 192 -> 192. VGPRCriticalLimit: 192 -> 192
define amdgpu_kernel void @test_waves_5(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

; waves-per-eu=6,6
; GFX942-40:  test_waves_6
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 40 -> 16. VGPRCriticalLimit: 40 -> 16
; GFX942-60:  test_waves_6
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 40 -> 24. VGPRCriticalLimit: 40 -> 24
; GFX942-80:  test_waves_6
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 40 -> 32. VGPRCriticalLimit: 40 -> 32
; GFX942-100: test_waves_6
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 40 -> 40. VGPRCriticalLimit: 40 -> 40

; GFX1250-40:  test_waves_6
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 160 -> 64. VGPRCriticalLimit: 160 -> 64
; GFX1250-60:  test_waves_6
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 160 -> 96. VGPRCriticalLimit: 160 -> 96
; GFX1250-80:  test_waves_6
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 160 -> 128. VGPRCriticalLimit: 160 -> 128
; GFX1250-100: test_waves_6
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 160 -> 160. VGPRCriticalLimit: 160 -> 160
define amdgpu_kernel void @test_waves_6(ptr addrspace(1) %out, ptr addrspace(1) %in) #2 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

; waves-per-eu=7,7
; GFX942-40:  test_waves_7
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 36 -> 15. VGPRCriticalLimit: 36 -> 15
; GFX942-60:  test_waves_7
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 36 -> 22. VGPRCriticalLimit: 36 -> 22
; GFX942-80:  test_waves_7
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 36 -> 29. VGPRCriticalLimit: 36 -> 29
; GFX942-100: test_waves_7
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 36 -> 36. VGPRCriticalLimit: 36 -> 36

; GFX1250-40:  test_waves_7
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 144 -> 58. VGPRCriticalLimit: 144 -> 58
; GFX1250-60:  test_waves_7
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 144 -> 87. VGPRCriticalLimit: 144 -> 87
; GFX1250-80:  test_waves_7
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 144 -> 116. VGPRCriticalLimit: 144 -> 116
; GFX1250-100: test_waves_7
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 144 -> 144. VGPRCriticalLimit: 144 -> 144
define amdgpu_kernel void @test_waves_7(ptr addrspace(1) %out, ptr addrspace(1) %in) #3 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

; waves-per-eu=8,8
; GFX942-40:  test_waves_8
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 32 -> 13. VGPRCriticalLimit: 32 -> 13
; GFX942-60:  test_waves_8
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 32 -> 20. VGPRCriticalLimit: 32 -> 20
; GFX942-80:  test_waves_8
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 32 -> 26. VGPRCriticalLimit: 32 -> 26
; GFX942-100: test_waves_8
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 32 -> 32. VGPRCriticalLimit: 32 -> 32

; GFX1250-40:  test_waves_8
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 128 -> 52. VGPRCriticalLimit: 128 -> 52
; GFX1250-60:  test_waves_8
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 128 -> 77. VGPRCriticalLimit: 128 -> 77
; GFX1250-80:  test_waves_8
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 128 -> 103. VGPRCriticalLimit: 128 -> 103
; GFX1250-100: test_waves_8
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 128 -> 128. VGPRCriticalLimit: 128 -> 128
define amdgpu_kernel void @test_waves_8(ptr addrspace(1) %out, ptr addrspace(1) %in) #4 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

; waves-per-eu=9,9
; gfx942 can only achieve max 8 waves (512 VGPRs / 64 per wave), so VGPRCriticalLimit is clamped
; GFX942-40:  test_waves_9
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 256 -> 103. VGPRCriticalLimit: 64 -> 26
; GFX942-60:  test_waves_9
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 256 -> 154. VGPRCriticalLimit: 64 -> 39
; GFX942-80:  test_waves_9
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 256 -> 205. VGPRCriticalLimit: 64 -> 52
; GFX942-100: test_waves_9
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 256 -> 256. VGPRCriticalLimit: 64 -> 64

; GFX1250-40:  test_waves_9
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 112 -> 45. VGPRCriticalLimit: 112 -> 45
; GFX1250-60:  test_waves_9
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 112 -> 68. VGPRCriticalLimit: 112 -> 68
; GFX1250-80:  test_waves_9
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 112 -> 90. VGPRCriticalLimit: 112 -> 90
; GFX1250-100: test_waves_9
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 112 -> 112. VGPRCriticalLimit: 112 -> 112
define amdgpu_kernel void @test_waves_9(ptr addrspace(1) %out, ptr addrspace(1) %in) #5 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

; waves-per-eu=10,10
; gfx942 can only achieve max 8 waves (512 VGPRs / 64 per wave), so VGPRCriticalLimit is clamped
; GFX942-40:  test_waves_10
; GFX942-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 256 -> 103. VGPRCriticalLimit: 64 -> 26
; GFX942-60:  test_waves_10
; GFX942-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 256 -> 154. VGPRCriticalLimit: 64 -> 39
; GFX942-80:  test_waves_10
; GFX942-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 256 -> 205. VGPRCriticalLimit: 64 -> 52
; GFX942-100: test_waves_10
; GFX942-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 256 -> 256. VGPRCriticalLimit: 64 -> 64

; GFX1250-40:  test_waves_10
; GFX1250-40:  Applied VGPR excess threshold 40%, VGPRExcessLimit: 96 -> 39. VGPRCriticalLimit: 96 -> 39
; GFX1250-60:  test_waves_10
; GFX1250-60:  Applied VGPR excess threshold 60%, VGPRExcessLimit: 96 -> 58. VGPRCriticalLimit: 96 -> 58
; GFX1250-80:  test_waves_10
; GFX1250-80:  Applied VGPR excess threshold 80%, VGPRExcessLimit: 96 -> 77. VGPRCriticalLimit: 96 -> 77
; GFX1250-100: test_waves_10
; GFX1250-100: Applied VGPR excess threshold 100%, VGPRExcessLimit: 96 -> 96. VGPRCriticalLimit: 96 -> 96
define amdgpu_kernel void @test_waves_10(ptr addrspace(1) %out, ptr addrspace(1) %in) #6 {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="1,128" }
attributes #1 = { "amdgpu-waves-per-eu"="5,5" "amdgpu-flat-work-group-size"="1,128" }
attributes #2 = { "amdgpu-waves-per-eu"="6,6" "amdgpu-flat-work-group-size"="1,128" }
attributes #3 = { "amdgpu-waves-per-eu"="7,7" "amdgpu-flat-work-group-size"="1,128" }
attributes #4 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="1,128" }
attributes #5 = { "amdgpu-waves-per-eu"="9,9" "amdgpu-flat-work-group-size"="1,128" }
attributes #6 = { "amdgpu-waves-per-eu"="10,10" "amdgpu-flat-work-group-size"="1,128" }
