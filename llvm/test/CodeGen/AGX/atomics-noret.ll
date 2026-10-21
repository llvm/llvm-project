; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; A device atomic whose old value is unused writes no register ("_") and
; needs no wait.
; CHECK-LABEL: counter:
; CHECK:         atomic add, 0, _, u0_u1, 0, signed, r{{[0-9]+}}, {{[01]}}
; CHECK-NEXT:    stop
define agx_kernel void @counter(ptr addrspace(1) %p, i32 %v) {
  %old = atomicrmw add ptr addrspace(1) %p, i32 %v monotonic
  ret void
}

; CHECK-LABEL: max_unused:
; CHECK:         atomic imax, 0, _,
; CHECK-NOT:     wait
; CHECK:         stop
define agx_kernel void @max_unused(ptr addrspace(1) %p, i32 %v) {
  %old = atomicrmw max ptr addrspace(1) %p, i32 %v monotonic
  ret void
}

; A used result still gets its register and its wait.
; CHECK-LABEL: used:
; CHECK:         atomic add, 0, r{{[0-9]+}},
; CHECK-NEXT:    wait 0
define agx_kernel void @used(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %v) {
  %old = atomicrmw add ptr addrspace(1) %p, i32 %v monotonic
  store i32 %old, ptr addrspace(1) %o
  ret void
}

; The G14 float add without a used result writes no register either
; (red.add.f32, ptx-re-blockers B24).
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=FADD
; FADD-LABEL: fsum:
; FADD:         atomic sub, 0, _, u0_u1, 0, signed, r{{[0-9]+}}, 1
; FADD-NEXT:    stop
define agx_kernel void @fsum(ptr addrspace(1) %p, float %v) {
  %old = atomicrmw fadd ptr addrspace(1) %p, float %v monotonic
  ret void
}
