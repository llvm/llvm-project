; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; Fences lower to the barriers measured with cross-core litmus tests on G14C
; (M2 Max): memory_barrier 1, 2, 9 is f596, 0, 3, 3 is f533 (store release),
; 3, 1, 9 is f59d (acquire, after wait 0). Threadgroup scope keeps f596;
; sequential consistency uses Mesa's seven-barrier sequence. Ordered atomics
; are monotonic accesses between the fences of their scope.

; CHECK-LABEL: release_agent:
; CHECK:       memory_barrier 0, 3, 3
; CHECK-NEXT:  stop
define agx_kernel void @release_agent() {
  fence syncscope("agent") release
  ret void
}

; CHECK-LABEL: acquire_agent:
; CHECK:       wait 0
; CHECK-NEXT:  memory_barrier 3, 1, 9
; CHECK-NEXT:  stop
define agx_kernel void @acquire_agent() {
  fence syncscope("agent") acquire
  ret void
}

; CHECK-LABEL: acq_rel_system:
; CHECK:       memory_barrier 0, 3, 3
; CHECK-NEXT:  wait 0
; CHECK-NEXT:  memory_barrier 3, 1, 9
; CHECK-NEXT:  stop
define agx_kernel void @acq_rel_system() {
  fence acq_rel
  ret void
}

; CHECK-LABEL: workgroup:
; CHECK:       memory_barrier 1, 2, 9
; CHECK-NEXT:  stop
define agx_kernel void @workgroup() {
  fence syncscope("workgroup") seq_cst
  ret void
}

; CHECK-LABEL: single_thread:
; CHECK-NOT:   memory_barrier
; CHECK:       stop
define agx_kernel void @single_thread() {
  fence syncscope("singlethread") seq_cst
  ret void
}

; A release atomic is preceded by the release barrier, an acquire one is
; followed by the acquire.
; CHECK-LABEL: release_atomic:
; CHECK:       memory_barrier 0, 3, 3
; CHECK:       atomic add
; CHECK-NOT:   memory_barrier
; CHECK:       stop
define agx_kernel void @release_atomic(ptr addrspace(1) %p) {
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("agent") release
  ret void
}

; CHECK-LABEL: acquire_atomic:
; CHECK-NOT:   memory_barrier
; CHECK:       atomic add
; CHECK:       wait 0
; CHECK-NEXT:  memory_barrier 3, 1, 9
define agx_kernel void @acquire_atomic(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("agent") acquire
  store i32 %old, ptr addrspace(1) %o
  ret void
}
