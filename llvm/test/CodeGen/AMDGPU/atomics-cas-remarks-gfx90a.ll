; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs --pass-remarks=atomic-expand \
; RUN:      %s -o - 2>&1 | FileCheck %s --check-prefix=GFX90A-CAS

; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at system memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at workgroup memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at wavefront memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at singlethread memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at one-as memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at agent-one-as memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at workgroup-one-as memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at wavefront-one-as memory scope
; GFX90A-CAS: A compare and swap loop was generated for an atomic fadd operation at singlethread-one-as memory scope

; GFX90A-CAS-LABEL: atomic_add_cas:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_agent:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_agent(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("agent") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_workgroup:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_workgroup(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("workgroup") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_wavefront:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_wavefront(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("wavefront") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_singlethread:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_singlethread(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("singlethread") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_one_as:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_one_as(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("one-as") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_agent_one_as:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_agent_one_as(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("agent-one-as") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_workgroup_one_as:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_workgroup_one_as(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("workgroup-one-as") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_wavefront_one_as:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_wavefront_one_as(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("wavefront-one-as") monotonic, align 4
  ret void
}

; GFX90A-CAS-LABEL: atomic_add_cas_singlethread_one_as:
; GFX90A-CAS: flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-CAS: s_cbranch_execnz
define dso_local void @atomic_add_cas_singlethread_one_as(ptr %p, float %q) {
entry:
  %ret = atomicrmw fadd ptr %p, float %q syncscope("singlethread-one-as") monotonic, align 4
  ret void
}
