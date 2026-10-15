; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<unknown=1>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<max-bytes=invalid>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<min-align=3>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<only-candidate=-2>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=BAD-CANDIDATE
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-lds-buffering<mode=unknown>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=BAD-MODE

; UNKNOWN: amdgpu-lds-buffering: invalid AMDGPU LDS buffering pass parameter 'unknown=1'
; INVALID: amdgpu-lds-buffering: invalid AMDGPU LDS buffering max-bytes 'invalid'
; BAD-ALIGN: amdgpu-lds-buffering: invalid AMDGPU LDS buffering min-align '3'
; BAD-CANDIDATE: amdgpu-lds-buffering: invalid AMDGPU LDS buffering only-candidate '-2'
; BAD-MODE: amdgpu-lds-buffering: invalid AMDGPU LDS buffering mode 'unknown'

define amdgpu_kernel void @kernel() {
  ret void
}
