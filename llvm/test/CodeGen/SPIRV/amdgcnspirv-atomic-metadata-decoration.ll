; Without -spirv-preserve-auxdata, AMDGPU atomic metadata must not appear
; as UserSemantic decorations or any other form in the SPIR-V output.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: amdgpu.no.fine.grained.memory
; CHECK-NOT: amdgpu.no.remote.memory
; CHECK-NOT: amdgpu.ignore.denormal.mode
; CHECK: %[[#Add:]] = OpAtomicIAdd
; CHECK: %[[#FAdd:]] = OpAtomicFAddEXT
; CHECK: %[[#Xchg:]] = OpAtomicExchange

define spir_func void @foo(ptr addrspace(1) %p) {
entry:
  %atomic.add = atomicrmw add ptr addrspace(1) %p, i32 1 seq_cst, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0
  %atomic.fadd = atomicrmw fadd ptr addrspace(1) %p, float 1.0 seq_cst, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0, !amdgpu.ignore.denormal.mode !0
  %atomic.xchg = atomicrmw xchg ptr addrspace(1) %p, i32 1 seq_cst, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
