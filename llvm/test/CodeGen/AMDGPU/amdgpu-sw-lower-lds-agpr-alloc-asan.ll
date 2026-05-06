; RUN: opt < %s -passes=amdgpu-sw-lower-lds -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a \
; RUN:   | FileCheck %s --check-prefix=GFX90A
; RUN: opt < %s -passes=amdgpu-sw-lower-lds -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 \
; RUN:   | FileCheck %s --check-prefix=NOGFX90A

; Test that amdgpu-agpr-alloc is propagated to __asan_malloc_impl,
; __asan_free_impl, and __asan_poison_region only on gfx90a targets.

@lds_1 = internal addrspace(3) global i32 poison, align 4

define amdgpu_kernel void @k0() sanitize_address #0 {
  store i32 1, ptr addrspace(3) @lds_1, align 4
  ret void
}

; GFX90A: declare i64 @__asan_malloc_impl(i64, i64) #[[AGPR_ATTR:[0-9]+]]
; GFX90A: declare void @__asan_poison_region(i64, i64) #[[AGPR_ATTR]]
; GFX90A: declare void @__asan_free_impl(i64, i64) #[[AGPR_ATTR]]

; GFX90A: attributes #[[AGPR_ATTR]] = { "amdgpu-agpr-alloc"="0" }

; NOGFX90A-NOT: declare i64 @__asan_malloc_impl{{.*}}"amdgpu-agpr-alloc"
; NOGFX90A-NOT: declare void @__asan_poison_region{{.*}}"amdgpu-agpr-alloc"
; NOGFX90A-NOT: declare void @__asan_free_impl{{.*}}"amdgpu-agpr-alloc"

attributes #0 = { sanitize_address "amdgpu-agpr-alloc"="0" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"nosanitize_address", i32 1}

