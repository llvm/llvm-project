; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
; REQUIRES: amdgpu-registered-target
;
; AMDGPU runs the register allocator more than once per function. Check that
; each function gets a single context record.
;
; RUN: llc -o /dev/null -mtriple=amdgpu9.0a-amd-amdhsa \
; RUN:   -regalloc-enable-advisor=development -mlregalloc-num-allocatable-regs=256 \
; RUN:   -regalloc-training-log=%t1 < %s
; RUN: grep -a '^{"\(context\|observation\|outcome\)"' %t1 \
; RUN:   | FileCheck %s --check-prefix=EVICT --match-full-lines

; RUN: llc -o /dev/null -mtriple=amdgpu9.0a-amd-amdhsa \
; RUN:   -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t2 < %s
; RUN: grep -a '^{"\(context\|observation\|outcome\)"' %t2 \
; RUN:   | FileCheck %s --check-prefix=PRIO --match-full-lines

; EVICT:      {"context":"{__unnamed_0}"}
; PRIO:      {"context":"{__unnamed_0}"}
; PRIO-NEXT: {"observation":0}
; PRIO-NEXT: {"outcome":0}
; PRIO-NEXT: {"observation":1}
; PRIO-NEXT: {"outcome":1}
; PRIO-NEXT: {"observation":2}
; PRIO-NEXT: {"outcome":2}
; PRIO-NEXT: {"observation":3}
define amdgpu_kernel void @0(ptr addrspace(1) %p) {
  %v = load volatile i32, ptr addrspace(1) %p
  store volatile i32 %v, ptr addrspace(1) %p
  ret void
}

; EVICT-NEXT: {"context":"{__unnamed_1}"}
; PRIO-NEXT: {"context":"{__unnamed_1}"}
; PRIO-NEXT: {"observation":0}
; PRIO-NEXT: {"outcome":0}
; PRIO-NEXT: {"observation":1}
; PRIO-NEXT: {"outcome":1}
; PRIO-NEXT: {"observation":2}
; PRIO-NEXT: {"outcome":2}
; PRIO-NEXT: {"observation":3}
define amdgpu_kernel void @1(ptr addrspace(1) %p) {
  %v = load volatile i32, ptr addrspace(1) %p
  store volatile i32 %v, ptr addrspace(1) %p
  ret void
}

; EVICT-NEXT: {"context":"f"}
; PRIO-NEXT: {"context":"f"}
; PRIO-NEXT: {"observation":0}
; PRIO-NEXT: {"outcome":0}
; PRIO-NEXT: {"observation":1}
; PRIO-NEXT: {"outcome":1}
; PRIO-NEXT: {"observation":2}
; PRIO-NEXT: {"outcome":2}
; PRIO-NEXT: {"observation":3}
; EVICT-NOT:  {{.}}
; PRIO-NOT:  {{.}}
define amdgpu_kernel void @f(ptr addrspace(1) %p) {
  %v = load volatile i32, ptr addrspace(1) %p
  store volatile i32 %v, ptr addrspace(1) %p
  ret void
}
