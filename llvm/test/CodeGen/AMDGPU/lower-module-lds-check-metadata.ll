; RUN: llc -mcpu=gfx906 -o - < %s | FileCheck --check-prefix=CHECK %s
target triple = "amdgcn-amd-amdhsa"

; Check the group segment has size 4, not zero.
; CHECK:       .amdhsa_kernel __device_start
; CHECK:       .amdhsa_group_segment_fixed_size 4
; CHECK:       .end_amdhsa_kernel

@global_barrier_state = hidden addrspace(3) global i32 undef, align 4

define i32 @rw() #0 {
entry:
  %0 = atomicrmw add ptr addrspace(3) @global_barrier_state, i32 1 acq_rel, align 4
  ret i32 %0
}

define amdgpu_kernel void @__device_start() {
entry:
  %0 = call i32 @rw()
  ret void
}

attributes #0 = { noinline  }
