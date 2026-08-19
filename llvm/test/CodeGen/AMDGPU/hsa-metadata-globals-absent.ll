; Code objects with nothing to report carry no amdhsa.globals key at all, so
; the note is unchanged for builds without sanitizer instrumentation.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o /dev/null < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

@scalar = addrspace(1) global i32 0, align 4
@array = addrspace(1) global [64 x float] zeroinitializer, align 4
@ro = addrspace(4) global i64 0, align 8

define amdgpu_kernel void @kern() {
  ret void
}

; CHECK-NOT: amdhsa.globals
; CHECK:     amdhsa.kernels:
; CHECK-NOT: amdhsa.globals
