; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

define void @valid_amdgpu_waves_per_eu_range() "amdgpu-waves-per-eu"="2,4" {
  ret void
}

define void @valid_amdgpu_waves_per_eu_min_only() "amdgpu-waves-per-eu"="2" {
  ret void
}

define void @valid_amdgpu_waves_per_eu_max_only() "amdgpu-waves-per-eu"="0,4" {
  ret void
}

; CHECK: minimum for 'amdgpu-waves-per-eu' must be integer: x
define void @invalid_amdgpu_waves_per_eu_min_nan() "amdgpu-waves-per-eu"="x" {
  ret void
}

; CHECK: maximum for 'amdgpu-waves-per-eu' must be integer: x
define void @invalid_amdgpu_waves_per_eu_max_nan() "amdgpu-waves-per-eu"="0,x" {
  ret void
}

; CHECK: minimum for 'amdgpu-waves-per-eu' must be non-zero when maximum is not provided
define void @invalid_amdgpu_waves_per_eu_min_zero() "amdgpu-waves-per-eu"="0" {
  ret void
}

; CHECK: maximum for 'amdgpu-waves-per-eu' must be non-zero
define void @invalid_amdgpu_waves_per_eu_max_zero() "amdgpu-waves-per-eu"="2,0" {
  ret void
}

; CHECK: minimum must be less than or equal to maximum for 'amdgpu-waves-per-eu': 2 > 1
define void @invalid_amdgpu_waves_per_eu_max_lt_min() "amdgpu-waves-per-eu"="2,1" {
  ret void
}
