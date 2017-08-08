; RUN: not llvm-as < %s 2>&1 | FileCheck %s

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
; CHECK-NEXT: void (i32*)* @sret_cc_amdgpu_kernel
define amdgpu_kernel void @sret_cc_amdgpu_kernel(i32* sret %ptr) {
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
