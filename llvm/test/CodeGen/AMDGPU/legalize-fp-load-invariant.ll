; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs -stop-after=amdgpu-isel -o - %s | FileCheck -check-prefix=GCN %s

; Type legalization for illegal FP type results was dropping invariant
; and dereferenceable flags.

; GCN: BUFFER_LOAD_USHORT{{.*}} :: (dereferenceable invariant load (s16) from %ir.ptr, addrspace 4)
define half @legalize_f16_load_align2(ptr addrspace(4) dereferenceable(4) align(2) %ptr) {
  %load = load half, ptr addrspace(4) %ptr, !invariant.load !0
  %add = fadd half %load, 1.0
  ret half %add
}

; GCN: BUFFER_LOAD_USHORT{{.*}} :: (invariant load (s16) from %ir.ptr, addrspace 4)
define half @legalize_f16_load_align1(ptr addrspace(4) dereferenceable(4) align(1) %ptr) {
  %load = load half, ptr addrspace(4) %ptr, !invariant.load !0
  %add = fadd half %load, 1.0
  ret half %add
}

!0 = !{}
