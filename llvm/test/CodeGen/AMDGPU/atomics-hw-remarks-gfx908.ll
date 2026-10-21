; RUN: llc -mtriple=amdgpu9.08 -pass-remarks='si-lower|atomic-expand' -filetype=null %s 2>&1 | \
; RUN:   FileCheck --implicit-check-not=remark: %s

; No LDS f64 fadd instruction.
; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at system memory scope
define void @local_atomicrmw_fadd_f64__nortn(ptr addrspace(3) %ptr, double %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(3) %ptr, double %val seq_cst
  ret void
}

; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define void @global_atomicrmw_fadd_f32_agent__nortn(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4
  ret void
}

; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define float @global_atomicrmw_fadd_f32_agent__rtn(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4
  ret float %ret
}

; Denormals are not handled.
; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_fine_grained_memory(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_remote_memory(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and denormals may be flushed (!atomic.ignore.denormal.mode)
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_fine_grained_memory__atomic_ignore_denormal_mode(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0, !atomic.ignore.denormal.mode !0
  ret void
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and denormals may be flushed (!atomic.ignore.denormal.mode)
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_remote_memory__amdgpu_no_fine_grained_memory__atomic_ignore_denormal_mode(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0, !amdgpu.no.fine.grained.memory !0, !atomic.ignore.denormal.mode !0
  ret void
}

; Remote memory is insufficient without agent scope fine-grained remote memory
; atomics.
; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_remote_memory__atomic_ignore_denormal_mode(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0, !atomic.ignore.denormal.mode !0
  ret void
}

; No f32 rtn instruction.
; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define float @global_atomicrmw_fadd_f32_agent__rtn__amdgpu_no_fine_grained_memory__atomic_ignore_denormal_mode(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0, !atomic.ignore.denormal.mode !0
  ret float %ret
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at system scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and denormals may be flushed (!atomic.ignore.denormal.mode)
define void @global_atomicrmw_fadd_f32_system__nortn__amdgpu_no_fine_grained_memory__atomic_ignore_denormal_mode(ptr addrspace(1) %ptr, float %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val monotonic, align 4, !amdgpu.no.fine.grained.memory !0, !atomic.ignore.denormal.mode !0
  ret void
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_fine_grained_memory__ftz(ptr addrspace(1) %ptr, float %val) #1 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_remote_memory__ftz(ptr addrspace(1) %ptr, float %val) #1 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
define void @global_atomicrmw_fadd_f32_agent__nortn__amdgpu_no_remote_memory__amdgpu_no_fine_grained_memory__ftz(ptr addrspace(1) %ptr, float %val) #1 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0, !amdgpu.no.fine.grained.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory)
define void @global_atomicrmw_fadd_v2f16_agent__nortn__amdgpu_no_fine_grained_memory(ptr addrspace(1) %ptr, <2 x half> %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, <2 x half> %val syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define void @global_atomicrmw_fadd_v2f16_agent__nortn__amdgpu_no_remote_memory(ptr addrspace(1) %ptr, <2 x half> %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, <2 x half> %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0
  ret void
}

; CHECK: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory)
define void @global_atomicrmw_fadd_v2f16_agent__nortn__amdgpu_no_fine_grained_memory__amdgpu_no_remote_memory(ptr addrspace(1) %ptr, <2 x half> %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, <2 x half> %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0, !amdgpu.no.fine.grained.memory !0
  ret void
}

; No v2f16 rtn instruction.
; CHECK: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at agent memory scope
define <2 x half> @global_atomicrmw_fadd_v2f16_agent__rtn__amdgpu_no_fine_grained_memory__amdgpu_no_remote_memory(ptr addrspace(1) %ptr, <2 x half> %val) #0 {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, <2 x half> %val syncscope("agent") monotonic, align 4, !amdgpu.no.remote.memory !0, !amdgpu.no.fine.grained.memory !0
  ret <2 x half> %ret
}

attributes #0 = { denormal_fpenv(ieee) }
attributes #1 = { denormal_fpenv(float: preservesign) }

!0 = !{}
