; REQUIRES: have_tflite
; REQUIRES: amdgpu-registered-target
;
; RUN: llc -o /dev/null -mtriple=amdgpu9.0a-amd-amdhsa \
; RUN:   -regalloc-enable-advisor=development -mlregalloc-num-allocatable-regs=256 \
; RUN:   -regalloc-training-log=%t1 < %s
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t1 > %t1.readable
; RUN: FileCheck --input-file %t1.readable %s

; RUN: llc -o /dev/null -mtriple=amdgpu9.0a-amd-amdhsa \
; RUN:   -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t2 < %s
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t2 > %t2.readable
; RUN: FileCheck --input-file %t2.readable %s

; CHECK: context: f
; CHECK: reward: 25.0
; CHECK-NOT: observation:

define void @f(ptr addrspace(1) %p) #0 {
  %a0 = load volatile i32, ptr addrspace(1) %p
  %a1 = load volatile i32, ptr addrspace(1) %p
  %a2 = load volatile i32, ptr addrspace(1) %p
  store volatile i32 %a0, ptr addrspace(1) %p
  store volatile i32 %a1, ptr addrspace(1) %p
  store volatile i32 %a2, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-num-vgpr"="2" }
