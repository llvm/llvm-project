; REQUIRES: amdgpu-registered-target
; RUN: opt < %S/chr-uniformity-profile.ll -mtriple=amdgpu9.50-amd-amdhsa -passes='default<O3>' -S | FileCheck %S/chr-uniformity-profile.ll

; Run the shared scope checks with the AMDGPU O3 pipeline. Keep this separately
; gated so the target-independent checks also run in builds without AMDGPU.
