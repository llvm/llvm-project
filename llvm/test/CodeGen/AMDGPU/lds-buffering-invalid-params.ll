; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<unknown=1>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<max-bytes=invalid>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=INVALID

; UNKNOWN: amdgpu-lds-buffering: invalid AMDGPU LDS buffering pass parameter 'unknown=1'
; INVALID: amdgpu-lds-buffering: invalid AMDGPU LDS buffering max-bytes 'invalid'

define amdgpu_kernel void @kernel() {
  ret void
}
