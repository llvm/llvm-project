; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a -passes=amdgpu-attributor %s | FileCheck %s

; Test that AAAMDGPUMinAGPRAlloc uses a pessimistic fixpoint for functions with
; sanitizer attributes, so amdgpu-agpr-alloc=0 is not inferred for them.
; Sanitizer runtime calls (e.g. __asan_malloc_impl) are introduced after the
; Attributor runs and may use AGPRs, so the Attributor cannot safely infer zero.

; Capture attribute group numbers from the define lines (before the attributes
; section). All CHECK-LABELs must precede the attribute section checks below.

; CHECK-LABEL: define amdgpu_kernel void @kernel_no_sanitizer(
; CHECK-SAME: ) #[[NO_SAN:[0-9]+]] {
define amdgpu_kernel void @kernel_no_sanitizer() {
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_sanitize_address(
; CHECK-SAME: ) #[[ASAN:[0-9]+]] {
define amdgpu_kernel void @kernel_sanitize_address() sanitize_address {
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_sanitize_memory(
; CHECK-SAME: ) #[[MSAN:[0-9]+]] {
define amdgpu_kernel void @kernel_sanitize_memory() sanitize_memory {
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_sanitize_thread(
; CHECK-SAME: ) #[[TSAN:[0-9]+]] {
define amdgpu_kernel void @kernel_sanitize_thread() sanitize_thread {
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_sanitize_hwaddress(
; CHECK-SAME: ) #[[HWASAN:[0-9]+]] {
define amdgpu_kernel void @kernel_sanitize_hwaddress() sanitize_hwaddress {
  ret void
}

; A non-sanitized kernel calling a sanitized callee. The callee's pessimistic
; fixpoint propagates up, so the caller must not get amdgpu-agpr-alloc=0 either.
define void @sanitized_callee() sanitize_address {
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @kernel_calls_sanitized_callee(
; CHECK-SAME: ) #[[CALLER:[0-9]+]] {
define amdgpu_kernel void @kernel_calls_sanitized_callee() {
  call void @sanitized_callee()
  ret void
}

; Control: no sanitizer -- amdgpu-agpr-alloc=0 must be inferred.
; CHECK: attributes #[[NO_SAN]] = { "amdgpu-agpr-alloc"="0"

; Sanitizer functions: amdgpu-agpr-alloc=0 must NOT appear.
; CHECK: attributes #[[ASAN]] = { sanitize_address
; CHECK-NOT: "amdgpu-agpr-alloc"="0"
; CHECK: attributes #[[MSAN]] = { sanitize_memory
; CHECK-NOT: "amdgpu-agpr-alloc"="0"
; CHECK: attributes #[[TSAN]] = { sanitize_thread
; CHECK-NOT: "amdgpu-agpr-alloc"="0"
; CHECK: attributes #[[HWASAN]] = { sanitize_hwaddress
; CHECK-NOT: "amdgpu-agpr-alloc"="0"

; Caller of a sanitized callee: also no amdgpu-agpr-alloc=0.
; CHECK: attributes #[[CALLER]] = {
; CHECK-NOT: "amdgpu-agpr-alloc"="0"

