; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK: OpDecorate %[[#Add:]] UserSemantic "amdgpu.no.fine.grained.memory"
; CHECK-NEXT: OpDecorate %[[#Add]] UserSemantic "amdgpu.no.remote.memory"
; CHECK-NEXT: OpDecorate %[[#FAdd:]] UserSemantic "amdgpu.no.fine.grained.memory"
; CHECK-NEXT: OpDecorate %[[#FAdd]] UserSemantic "amdgpu.no.remote.memory"
; CHECK-NEXT: OpDecorate %[[#FAdd]] UserSemantic "amdgpu.ignore.denormal.mode"
; CHECK: %[[#Add]] = OpAtomicIAdd
; CHECK: %[[#FAdd]] = OpAtomicFAddEXT

define spir_kernel void @foo(ptr addrspace(1) %p) {
entry:
  %atomic.add = atomicrmw add ptr addrspace(1) %p, i32 1 seq_cst, !amdgpu.no.fine.grained.memory !1, !amdgpu.no.remote.memory !1
  %atomic.fadd = atomicrmw fadd ptr addrspace(1) %p, float 1.0 seq_cst, !amdgpu.no.fine.grained.memory !1, !amdgpu.no.remote.memory !1, !amdgpu.ignore.denormal.mode !1
  ret void
}

!0 = !{i32 5, i32 6}
!1 = !{}
