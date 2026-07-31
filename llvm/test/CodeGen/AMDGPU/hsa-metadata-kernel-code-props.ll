; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc --amdgpu-xnack=false -mtriple=amdgpu8.03-amd-amdhsa -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc --amdgpu-xnack=false -mtriple=amdgpu9.00-amd-amdhsa -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc --amdgpu-xnack=false -mtriple=amdgpu10.10-amd-amdhsa -enable-misched=0 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,WAVE32 %s

@var = addrspace(1) global float 0.0

; CHECK: ---
; CHECK:  amdhsa.kernels:

; CHECK:   - .args:
; CHECK:     .group_segment_fixed_size: 0
; CHECK:     .kernarg_segment_align: 8
; CHECK:     .kernarg_segment_size: 24
; CHECK:     .max_flat_workgroup_size: 1024
; CHECK:     .name:           test
; CHECK:     .private_segment_fixed_size: 0
; CHECK:     .sgpr_count:     10
; CHECK:     .symbol:         test.kd
; CHECK:     .vgpr_count:     {{3|6}}
; WAVE64:    .wavefront_size: 64
; WAVE32:    .wavefront_size: 32
define amdgpu_kernel void @test(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) "amdgpu-no-implicitarg-ptr" "amdgpu-no-flat-scratch-init" {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:   - .args:
; CHECK:     .max_flat_workgroup_size: 256
define amdgpu_kernel void @test_max_flat_workgroup_size(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #2 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 1

attributes #2 = { "amdgpu-flat-work-group-size"="1,256" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
