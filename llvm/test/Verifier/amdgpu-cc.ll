; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A5"

; CHECK: Calling convention requires void return type
; CHECK-NEXT: ptr @nonvoid_cc_amdgpu_kernel
define amdgpu_kernel i32 @nonvoid_cc_amdgpu_kernel() {
  ret i32 0
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_kernel
define amdgpu_kernel void @varargs_amdgpu_kernel(...) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: ptr @sret_cc_amdgpu_kernel_as0
define amdgpu_kernel void @sret_cc_amdgpu_kernel_as0(ptr sret(i32) %ptr) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: ptr @sret_cc_amdgpu_kernel
define amdgpu_kernel void @sret_cc_amdgpu_kernel(ptr addrspace(5) sret(i32) %ptr) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_vs
define amdgpu_vs void @varargs_amdgpu_vs(...) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_gs
define amdgpu_gs void @varargs_amdgpu_gs(...) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_ps
define amdgpu_ps void @varargs_amdgpu_ps(...) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_cs
define amdgpu_cs void @varargs_amdgpu_cs(...) {
  ret void
}

; CHECK: Calling convention requires void return type
; CHECK-NEXT: ptr @nonvoid_cc_spir_kernel
define spir_kernel i32 @nonvoid_cc_spir_kernel() {
  ret i32 0
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_spir_kernel
define spir_kernel void @varargs_spir_kernel(...) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_kernel
define amdgpu_kernel void @byval_cc_amdgpu_kernel(ptr addrspace(5) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_as1_cc_amdgpu_kernel
define amdgpu_kernel void @byval_as1_cc_amdgpu_kernel(ptr addrspace(1) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_as0_cc_amdgpu_kernel
define amdgpu_kernel void @byval_as0_cc_amdgpu_kernel(ptr byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_vs
define amdgpu_vs void @byval_cc_amdgpu_vs(ptr addrspace(5) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_hs
define amdgpu_hs void @byval_cc_amdgpu_hs(ptr addrspace(5) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_gs
define amdgpu_gs void @byval_cc_amdgpu_gs(ptr addrspace(5) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_ps
define amdgpu_ps void @byval_cc_amdgpu_ps(ptr addrspace(5) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_cs
define amdgpu_cs void @byval_cc_amdgpu_cs(ptr addrspace(5) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows preallocated
; CHECK-NEXT: ptr @preallocated_as0_cc_amdgpu_kernel
define amdgpu_kernel void @preallocated_as0_cc_amdgpu_kernel(ptr preallocated(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows inalloca
; CHECK-NEXT: ptr @inalloca_as0_cc_amdgpu_kernel
define amdgpu_kernel void @inalloca_as0_cc_amdgpu_kernel(ptr inalloca(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows stack byref
; CHECK-NEXT: ptr @byref_as5_cc_amdgpu_kernel
define amdgpu_kernel void @byref_as5_cc_amdgpu_kernel(ptr addrspace(5) byref(i32) %ptr) {
  ret void
}

; CHECK: Calling convention requires void return type
; CHECK-NEXT: ptr @nonvoid_cc_amdgpu_cs_chain
define amdgpu_cs_chain i32 @nonvoid_cc_amdgpu_cs_chain() {
  ret i32 0
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_cs_chain
define amdgpu_cs_chain void @varargs_amdgpu_cs_chain(...) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: ptr @sret_cc_amdgpu_cs_chain_as0
define amdgpu_cs_chain void @sret_cc_amdgpu_cs_chain_as0(ptr sret(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_cs_chain
define amdgpu_cs_chain void @byval_cc_amdgpu_cs_chain(ptr addrspace(1) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows stack byref
; CHECK-NEXT: ptr @byref_cc_amdgpu_cs_chain
define amdgpu_cs_chain void @byref_cc_amdgpu_cs_chain(ptr addrspace(5) byref(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows preallocated
; CHECK-NEXT: ptr @preallocated_cc_amdgpu_cs_chain
define amdgpu_cs_chain void @preallocated_cc_amdgpu_cs_chain(ptr preallocated(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows inalloca
; CHECK-NEXT: ptr @inalloca_cc_amdgpu_cs_chain
define amdgpu_cs_chain void @inalloca_cc_amdgpu_cs_chain(ptr inalloca(i32) %ptr) {
  ret void
}

; CHECK: Calling convention requires void return type
; CHECK-NEXT: ptr @nonvoid_cc_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve i32 @nonvoid_cc_amdgpu_cs_chain_preserve() {
  ret i32 0
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: ptr @varargs_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve void @varargs_amdgpu_cs_chain_preserve(...) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: ptr @sret_cc_amdgpu_cs_chain_preserve_as0
define amdgpu_cs_chain_preserve void @sret_cc_amdgpu_cs_chain_preserve_as0(ptr sret(i32) %ptr) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: ptr @sret_cc_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve void @sret_cc_amdgpu_cs_chain_preserve(ptr addrspace(5) sret(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: ptr @byval_cc_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve void @byval_cc_amdgpu_cs_chain_preserve(ptr addrspace(1) byval(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows stack byref
; CHECK-NEXT: ptr @byref_cc_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve void @byref_cc_amdgpu_cs_chain_preserve(ptr addrspace(5) byref(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows preallocated
; CHECK-NEXT: ptr @preallocated_cc_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve void @preallocated_cc_amdgpu_cs_chain_preserve(ptr preallocated(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows inalloca
; CHECK-NEXT: ptr @inalloca_cc_amdgpu_cs_chain_preserve
define amdgpu_cs_chain_preserve void @inalloca_cc_amdgpu_cs_chain_preserve(ptr inalloca(i32) %ptr) {
  ret void
}

declare amdgpu_cs_chain void @amdgpu_cs_chain_call_target()
declare amdgpu_cs_chain_preserve void @amdgpu_cs_chain_preserve_call_target()

define amdgpu_cs_chain void @cant_call_amdgpu_cs_chain_functions(ptr %f) {
  ; CHECK: Direct calls to amdgpu_cs_chain/amdgpu_cs_chain_preserve functions not allowed. Please use the @llvm.amdgpu.cs.chain intrinsic instead.
  ; CHECK-NEXT: call amdgpu_cs_chain
  call amdgpu_cs_chain void @amdgpu_cs_chain_call_target()

  ; CHECK: Direct calls to amdgpu_cs_chain/amdgpu_cs_chain_preserve functions not allowed. Please use the @llvm.amdgpu.cs.chain intrinsic instead.
  ; CHECK-NEXT: call amdgpu_cs_chain_preserve
  call amdgpu_cs_chain_preserve void @amdgpu_cs_chain_preserve_call_target()

  ; CHECK: Direct calls to amdgpu_cs_chain/amdgpu_cs_chain_preserve functions not allowed. Please use the @llvm.amdgpu.cs.chain intrinsic instead.
  ; CHECK-NEXT: call amdgpu_cs_chain
  call amdgpu_cs_chain void %f()

  ; CHECK: Direct calls to amdgpu_cs_chain/amdgpu_cs_chain_preserve functions not allowed. Please use the @llvm.amdgpu.cs.chain intrinsic instead.
  ; CHECK-NEXT: call amdgpu_cs_chain
  call amdgpu_cs_chain_preserve void %f()

  ret void
}
