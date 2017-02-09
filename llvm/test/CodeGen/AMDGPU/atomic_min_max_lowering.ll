; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji -amdgpu-lower-opencl-atomic-builtins < %s | FileCheck %s


; FUNC-LABEL: @atomic_umin(
; CHECK: atomicrmw umin i32 addrspace(3)* %ptr, i32 4 syncscope(2) monotonic
define void @atomic_umin(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) #0 {
  %result =  tail call i32 @_Z10atomic_minPU3AS3Vjj(i32 addrspace(3)* %ptr, i32 4) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @atomic_smin(
; CHECK: atomicrmw min i32 addrspace(3)* %ptr, i32 4 syncscope(2) monotonic
define void @atomic_smin(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) #0 {
  %result =  tail call i32 @_Z10atomic_minPU3AS3Vii(i32 addrspace(3)* %ptr, i32 4) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @atomic_umax(
; CHECK: atomicrmw umax i32 addrspace(1)* %ptr, i32 4 syncscope(2) monotonic
define void @atomic_umax(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %result =  tail call i32 @_Z10atomic_maxPU3AS1Vjj(i32 addrspace(1)* %ptr, i32 4) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @atomic_smax(
; CHECK: atomicrmw max i32 addrspace(1)* %ptr, i32 4 syncscope(2) monotonic
define void @atomic_smax(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %result =  tail call i32 @_Z10atomic_maxPU3AS1Vii(i32 addrspace(1)* %ptr, i32 4) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}


declare i32 @_Z10atomic_minPU3AS3Vjj(i32 addrspace(3)*, i32)
declare i32 @_Z10atomic_minPU3AS3Vii(i32 addrspace(3)*, i32)
declare i32 @_Z10atomic_maxPU3AS1Vjj(i32 addrspace(1)*, i32)
declare i32 @_Z10atomic_maxPU3AS1Vii(i32 addrspace(1)*, i32)

attributes #0 = { nounwind }
