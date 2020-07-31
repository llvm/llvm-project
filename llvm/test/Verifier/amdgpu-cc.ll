; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A5"

; CHECK: Calling convention requires void return type
; CHECK-NEXT: i32 ()* @nonvoid_cc_amdgpu_kernel
define amdgpu_kernel i32 @nonvoid_cc_amdgpu_kernel() {
  ret i32 0
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: void (...)* @varargs_amdgpu_kernel
define amdgpu_kernel void @varargs_amdgpu_kernel(...) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: void (i32*)* @sret_cc_amdgpu_kernel_as0
define amdgpu_kernel void @sret_cc_amdgpu_kernel_as0(i32* sret %ptr) {
  ret void
}

; CHECK: Calling convention does not allow sret
; CHECK-NEXT: void (i32 addrspace(5)*)* @sret_cc_amdgpu_kernel
define amdgpu_kernel void @sret_cc_amdgpu_kernel(i32 addrspace(5)* sret %ptr) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: void (...)* @varargs_amdgpu_vs
define amdgpu_vs void @varargs_amdgpu_vs(...) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: void (...)* @varargs_amdgpu_gs
define amdgpu_gs void @varargs_amdgpu_gs(...) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: void (...)* @varargs_amdgpu_ps
define amdgpu_ps void @varargs_amdgpu_ps(...) {
  ret void
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: void (...)* @varargs_amdgpu_cs
define amdgpu_cs void @varargs_amdgpu_cs(...) {
  ret void
}

; CHECK: Calling convention requires void return type
; CHECK-NEXT: i32 ()* @nonvoid_cc_spir_kernel
define spir_kernel i32 @nonvoid_cc_spir_kernel() {
  ret i32 0
}

; CHECK: Calling convention does not support varargs or perfect forwarding!
; CHECK-NEXT: void (...)* @varargs_spir_kernel
define spir_kernel void @varargs_spir_kernel(...) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(5)*)* @byval_cc_amdgpu_kernel
define amdgpu_kernel void @byval_cc_amdgpu_kernel(i32 addrspace(5)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(1)*)* @byval_as1_cc_amdgpu_kernel
define amdgpu_kernel void @byval_as1_cc_amdgpu_kernel(i32 addrspace(1)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32*)* @byval_as0_cc_amdgpu_kernel
define amdgpu_kernel void @byval_as0_cc_amdgpu_kernel(i32* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(5)*)* @byval_cc_amdgpu_vs
define amdgpu_vs void @byval_cc_amdgpu_vs(i32 addrspace(5)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(5)*)* @byval_cc_amdgpu_hs
define amdgpu_hs void @byval_cc_amdgpu_hs(i32 addrspace(5)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(5)*)* @byval_cc_amdgpu_gs
define amdgpu_gs void @byval_cc_amdgpu_gs(i32 addrspace(5)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(5)*)* @byval_cc_amdgpu_ps
define amdgpu_ps void @byval_cc_amdgpu_ps(i32 addrspace(5)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows byval
; CHECK-NEXT: void (i32 addrspace(5)*)* @byval_cc_amdgpu_cs
define amdgpu_cs void @byval_cc_amdgpu_cs(i32 addrspace(5)* byval %ptr) {
  ret void
}

; CHECK: Calling convention disallows preallocated
; CHECK-NEXT: void (i32*)* @preallocated_as0_cc_amdgpu_kernel
define amdgpu_kernel void @preallocated_as0_cc_amdgpu_kernel(i32* preallocated(i32) %ptr) {
  ret void
}

; CHECK: Calling convention disallows inalloca
; CHECK-NEXT: void (i32*)* @inalloca_as0_cc_amdgpu_kernel
define amdgpu_kernel void @inalloca_as0_cc_amdgpu_kernel(i32* inalloca %ptr) {
  ret void
}

; CHECK: Calling convention disallows stack byref
; CHECK-NEXT: void (i32 addrspace(5)*)* @byref_as5_cc_amdgpu_kernel
define amdgpu_kernel void @byref_as5_cc_amdgpu_kernel(i32 addrspace(5)* byref(i32) %ptr) {
  ret void
}
