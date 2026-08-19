; Check that the instrumentation and the note agree end to end: asan attaches
; the declared size and the streamer reports it.

; RUN: opt -passes=asan -S < %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o /dev/null 2>&1 | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

@scalar = addrspace(1) global i32 7, align 4
@array = addrspace(1) global [64 x float] zeroinitializer, align 4

define amdgpu_kernel void @kern() sanitize_address {
  ret void
}

; The descriptor globals asan adds for its runtime carry no attachment, so the
; sequence ends here.

; CHECK:      amdhsa.globals:
; CHECK-NEXT:   - .name: scalar
; CHECK-NEXT:     .size: 4
; CHECK-NEXT:   - .name: array
; CHECK-NEXT:     .size: 256
; CHECK-NEXT: amdhsa.kernels:
