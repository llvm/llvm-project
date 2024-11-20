; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s -global-isel | FileCheck %s

; CHECK-LABEL: {{^}}signal_unknown_wgs:
; CHECK: s_barrier_signal
define amdgpu_kernel void @signal_unknown_wgs() {
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1) #0
  ret void
}

; CHECK-LABEL: {{^}}signal_flat_wgs_attr_32_128:
; CHECK: s_barrier_signal
define amdgpu_kernel void @signal_flat_wgs_attr_32_128() #1 {
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1) #0
  ret void
}

; CHECK-LABEL: {{^}}signal_flat_wgs_attr_32_64:
; CHECK: :
; CHECK-NEXT: ; wave barrier
; CHECK-NEXT: s_endpgm
define amdgpu_kernel void @signal_flat_wgs_attr_32_64() #2 {
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1) #0
  ret void
}


; CHECK-LABEL: {{^}}wait_unknown_wgs:
; CHECK: s_barrier_wait
define amdgpu_kernel void @wait_unknown_wgs() {
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1) #0
  ret void
}

; CHECK-LABEL: {{^}}wait_flat_wgs_attr_32_128:
; CHECK: s_barrier_wait
define amdgpu_kernel void @wait_flat_wgs_attr_32_128() #1 {
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1) #0
  ret void
}

; CHECK-LABEL: {{^}}wait_flat_wgs_attr_32_64:
; CHECK: :
; CHECK-NEXT: ; wave barrier
; CHECK-NEXT: s_endpgm
define amdgpu_kernel void @wait_flat_wgs_attr_32_64() #2 {
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1) #0
  ret void
}

declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #0
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #0

attributes #0 = { convergent nounwind }
attributes #1 = { nounwind "amdgpu-flat-work-group-size"="32,128" }
attributes #2 = { nounwind "amdgpu-flat-work-group-size"="16,32" }
