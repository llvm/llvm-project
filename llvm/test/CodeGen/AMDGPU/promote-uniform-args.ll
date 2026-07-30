; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=amdgpu-promote-uniform-args < %s | FileCheck %s

; A uniform pointer argument of an internal function, passed from a kernel,
; is promoted to inreg (SGPR) on both the definition and the call site.

; CHECK-LABEL: define internal fastcc void @callee_uniform(
; CHECK-SAME: ptr inreg {{.*}}%p
define internal fastcc void @callee_uniform(ptr %p, i32 %i) {
  %g = getelementptr float, ptr %p, i32 %i
  %v = load float, ptr %g
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_uniform(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_uniform(
; CHECK: call fastcc void @callee_uniform(ptr inreg %p, i32 %tid)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  call fastcc void @callee_uniform(ptr %p, i32 %tid)
  ret void
}

; A divergent pointer operand (derived from the workitem id) must NOT be
; promoted, because inreg would drop all but one lane's value.

; CHECK-LABEL: define internal fastcc void @callee_divergent(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_divergent(ptr %p) {
  %v = load float, ptr %p
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_divergent(ptr %base) {
; CHECK-LABEL: define amdgpu_kernel void @k_divergent(
; CHECK: call fastcc void @callee_divergent(ptr %pdiv)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %pdiv = getelementptr float, ptr %base, i32 %tid
  call fastcc void @callee_divergent(ptr %pdiv)
  ret void
}

; Private pointers name lane-private storage. Even if the pointer value itself
; is uniform, the callee must not learn that the pointee is wave-uniform.

; CHECK-LABEL: define internal fastcc void @callee_private(
; CHECK-SAME: ptr addrspace(5) %p
; CHECK-NOT: ptr addrspace(5) inreg
define internal fastcc void @callee_private(ptr addrspace(5) %p) {
  store i32 1, ptr addrspace(5) %p
  ret void
}

define amdgpu_kernel void @k_private() {
; CHECK-LABEL: define amdgpu_kernel void @k_private(
; CHECK: call fastcc void @callee_private(ptr addrspace(5) %a)
  %a = alloca i32, addrspace(5)
  call fastcc void @callee_private(ptr addrspace(5) %a)
  ret void
}

; A flat pointer derived from private memory carries the same risk.

; CHECK-LABEL: define internal fastcc void @callee_flat_private(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_flat_private(ptr %p) {
  store i32 1, ptr %p
  ret void
}

define amdgpu_kernel void @k_flat_private() {
; CHECK-LABEL: define amdgpu_kernel void @k_flat_private(
; CHECK: call fastcc void @callee_flat_private(ptr %f)
  %a = alloca i32, addrspace(5)
  %f = addrspacecast ptr addrspace(5) %a to ptr
  call fastcc void @callee_flat_private(ptr %f)
  ret void
}

; A flat pointer that may be private on one path of a phi/select must not be
; promoted, even though the other path is a benign global pointer.

; CHECK-LABEL: define internal fastcc void @callee_phi_private(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_phi_private(ptr %p) {
  store i32 1, ptr %p
  ret void
}

@gvar = addrspace(1) global i32 0

define amdgpu_kernel void @k_phi_private(i1 %c) {
; CHECK-LABEL: define amdgpu_kernel void @k_phi_private(
; CHECK: call fastcc void @callee_phi_private(ptr %sel)
  %a = alloca i32, addrspace(5)
  %fa = addrspacecast ptr addrspace(5) %a to ptr
  %fg = addrspacecast ptr addrspace(1) @gvar to ptr
  %sel = select i1 %c, ptr %fa, ptr %fg
  call fastcc void @callee_phi_private(ptr %sel)
  ret void
}

; A private pointer reached through a long getelementptr chain stays in the
; private address space, so the address-space check still blocks it.

; CHECK-LABEL: define internal fastcc void @callee_deep_private(
; CHECK-SAME: ptr addrspace(5) %p
; CHECK-NOT: ptr addrspace(5) inreg
define internal fastcc void @callee_deep_private(ptr addrspace(5) %p) {
  store i32 1, ptr addrspace(5) %p
  ret void
}

define amdgpu_kernel void @k_deep_private() {
; CHECK-LABEL: define amdgpu_kernel void @k_deep_private(
; CHECK: call fastcc void @callee_deep_private(ptr addrspace(5) %g12)
  %a = alloca [64 x i32], addrspace(5)
  %g1 = getelementptr i32, ptr addrspace(5) %a, i32 1
  %g2 = getelementptr i32, ptr addrspace(5) %g1, i32 1
  %g3 = getelementptr i32, ptr addrspace(5) %g2, i32 1
  %g4 = getelementptr i32, ptr addrspace(5) %g3, i32 1
  %g5 = getelementptr i32, ptr addrspace(5) %g4, i32 1
  %g6 = getelementptr i32, ptr addrspace(5) %g5, i32 1
  %g7 = getelementptr i32, ptr addrspace(5) %g6, i32 1
  %g8 = getelementptr i32, ptr addrspace(5) %g7, i32 1
  %g9 = getelementptr i32, ptr addrspace(5) %g8, i32 1
  %g10 = getelementptr i32, ptr addrspace(5) %g9, i32 1
  %g11 = getelementptr i32, ptr addrspace(5) %g10, i32 1
  %g12 = getelementptr i32, ptr addrspace(5) %g11, i32 1
  call fastcc void @callee_deep_private(ptr addrspace(5) %g12)
  ret void
}

; Callee-side guard: even a uniform flat pointer must not be promoted if the
; callee reinterprets it as private (scratch), because that extracts a
; lane-relative offset that is not wave-uniform.

; CHECK-LABEL: define internal fastcc void @callee_casts_private(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_casts_private(ptr %p) {
  %q = addrspacecast ptr %p to ptr addrspace(5)
  store i32 1, ptr addrspace(5) %q
  ret void
}

define amdgpu_kernel void @k_casts_private(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_casts_private(
; CHECK: call fastcc void @callee_casts_private(ptr %p)
  call fastcc void @callee_casts_private(ptr %p)
  ret void
}

; Positive control: a flat load/store through the argument is the common, safe
; use and must still be promoted.

; CHECK-LABEL: define internal fastcc void @callee_flat_load(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @callee_flat_load(ptr %p) {
  %v = load float, ptr %p
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_flat_load(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_flat_load(
; CHECK: call fastcc void @callee_flat_load(ptr inreg %p)
  call fastcc void @callee_flat_load(ptr %p)
  ret void
}

; SGPR dword budget (default 8): only the first four pointer arguments fit;
; the fifth is left in VGPRs.

; CHECK-LABEL: define internal fastcc void @callee_budget(
; CHECK-SAME: ptr inreg %p0, ptr inreg %p1, ptr inreg %p2, ptr inreg %p3, ptr %p4
define internal fastcc void @callee_budget(ptr %p0, ptr %p1, ptr %p2, ptr %p3,
                                           ptr %p4) {
  ret void
}

define amdgpu_kernel void @k_budget(ptr %p0, ptr %p1, ptr %p2, ptr %p3,
                                    ptr %p4) {
; CHECK-LABEL: define amdgpu_kernel void @k_budget(
; CHECK: call fastcc void @callee_budget(ptr inreg %p0, ptr inreg %p1, ptr inreg %p2, ptr inreg %p3, ptr %p4)
  call fastcc void @callee_budget(ptr %p0, ptr %p1, ptr %p2, ptr %p3, ptr %p4)
  ret void
}

; TTI cross-check: a flat load is NeverUniform in GCNTTI even when the address
; is wave-uniform, so the operand must not be promoted.

; CHECK-LABEL: define internal fastcc void @callee_tti_flatload(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_tti_flatload(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

@gptr = addrspace(1) global ptr null

define amdgpu_kernel void @k_tti_flatload() {
; CHECK-LABEL: define amdgpu_kernel void @k_tti_flatload(
; CHECK: call fastcc void @callee_tti_flatload(ptr %p)
  %flatg = addrspacecast ptr addrspace(1) @gptr to ptr
  %p = load ptr, ptr %flatg
  call fastcc void @callee_tti_flatload(ptr %p)
  ret void
}

; External linkage: all call sites are not necessarily visible, so the ABI
; must not be changed.

; CHECK-LABEL: define fastcc void @callee_external(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define fastcc void @callee_external(ptr %p) {
  %v = load float, ptr %p
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_external(ptr %p) {
  call fastcc void @callee_external(ptr %p)
  ret void
}

; Address-taken internal function: an indirect call we cannot see could pass a
; divergent value, so do not promote.

; CHECK-LABEL: define internal fastcc void @callee_addrtaken(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_addrtaken(ptr %p) {
  %v = load float, ptr %p
  store float %v, ptr %p
  ret void
}

@fnptr = global ptr null

define amdgpu_kernel void @k_addrtaken(ptr %p) {
  store ptr @callee_addrtaken, ptr @fnptr
  call fastcc void @callee_addrtaken(ptr %p)
  ret void
}

; Arguments with ABI-affecting attributes must not be promoted.

%struct.Foo = type { i32 }

; CHECK-LABEL: define internal fastcc void @callee_byref(
; CHECK-SAME: ptr byref(%struct.Foo) %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_byref(ptr byref(%struct.Foo) %p) {
  ret void
}

define amdgpu_kernel void @k_byref(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_byref(
; CHECK: call fastcc void @callee_byref(ptr byref(%struct.Foo) %p)
  call fastcc void @callee_byref(ptr byref(%struct.Foo) %p)
  ret void
}

; CHECK-LABEL: define internal fastcc ptr @callee_returned(
; CHECK-SAME: ptr returned %p
; CHECK-NOT: ptr inreg
define internal fastcc ptr @callee_returned(ptr returned %p) {
  ret ptr %p
}

define amdgpu_kernel void @k_returned(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_returned(
; CHECK: call fastcc ptr @callee_returned(ptr returned %p)
  %r = call fastcc ptr @callee_returned(ptr returned %p)
  store ptr %r, ptr %p
  ret void
}

; CHECK-LABEL: define internal fastcc void @callee_swiftasync(
; CHECK-SAME: ptr swiftasync %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_swiftasync(ptr swiftasync %p) {
  ret void
}

define amdgpu_kernel void @k_swiftasync(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_swiftasync(
; CHECK: call fastcc void @callee_swiftasync(ptr swiftasync %p)
  call fastcc void @callee_swiftasync(ptr swiftasync %p)
  ret void
}

; CHECK-LABEL: define internal fastcc void @callee_hidden(
; CHECK-SAME: ptr "amdgpu-hidden-argument" %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_hidden(ptr "amdgpu-hidden-argument" %p) {
  ret void
}

define amdgpu_kernel void @k_hidden(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_hidden(
; CHECK: call fastcc void @callee_hidden(ptr "amdgpu-hidden-argument" %p)
  call fastcc void @callee_hidden(ptr "amdgpu-hidden-argument" %p)
  ret void
}

; Non-pointer uniform arguments are out of scope for this pass.

; CHECK-LABEL: define internal fastcc void @callee_scalar(
; CHECK-SAME: i32 %n
; CHECK-NOT: i32 inreg
define internal fastcc void @callee_scalar(i32 %n, ptr %p) {
  store i32 %n, ptr %p
  ret void
}

define amdgpu_kernel void @k_scalar(i32 %n, ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_scalar(
; The pointer is still promoted, the scalar is not.
; CHECK: call fastcc void @callee_scalar(i32 %n, ptr inreg %p)
  call fastcc void @callee_scalar(i32 %n, ptr %p)
  ret void
}

; Mixed call sites: one uniform, one divergent. A single divergent operand must
; block promotion, since the definition is shared by all callers.

; CHECK-LABEL: define internal fastcc void @callee_mixed(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_mixed(ptr %p) {
  %v = load float, ptr %p
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_mixed_uniform(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_mixed_uniform(
; CHECK: call fastcc void @callee_mixed(ptr %p)
  call fastcc void @callee_mixed(ptr %p)
  ret void
}

define amdgpu_kernel void @k_mixed_divergent(ptr %base) {
; CHECK-LABEL: define amdgpu_kernel void @k_mixed_divergent(
; CHECK: call fastcc void @callee_mixed(ptr %pdiv)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %pdiv = getelementptr float, ptr %base, i32 %tid
  call fastcc void @callee_mixed(ptr %pdiv)
  ret void
}

; Multi-hop chain: uniformity propagates from the kernel through each internal
; function to a fixpoint, so every hop's closure pointer is promoted.

; CHECK-LABEL: define internal fastcc void @chain_leaf(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @chain_leaf(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

; CHECK-LABEL: define internal fastcc void @chain_mid(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @chain_mid(ptr %p) {
; CHECK: call fastcc void @chain_leaf(ptr inreg %p)
  call fastcc void @chain_leaf(ptr %p)
  ret void
}

; CHECK-LABEL: define internal fastcc void @chain_top(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @chain_top(ptr %p) {
; CHECK: call fastcc void @chain_mid(ptr inreg %p)
  call fastcc void @chain_mid(ptr %p)
  ret void
}

define amdgpu_kernel void @k_chain(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_chain(
; CHECK: call fastcc void @chain_top(ptr inreg %p)
  call fastcc void @chain_top(ptr %p)
  ret void
}

; Fixpoint must converge regardless of the order the functions appear in the
; module (callees defined before/after their callers) and across a diamond.

; CHECK-LABEL: define internal fastcc void @scram_a(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @scram_a(ptr %p) {
  call fastcc void @scram_b(ptr %p)
  ret void
}

; CHECK-LABEL: define internal fastcc void @scram_c(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @scram_c(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

; CHECK-LABEL: define internal fastcc void @scram_b(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @scram_b(ptr %p) {
  call fastcc void @scram_c(ptr %p)
  ret void
}

define amdgpu_kernel void @k_scram(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_scram(
; CHECK: call fastcc void @scram_a(ptr inreg %p)
  call fastcc void @scram_a(ptr %p)
  ret void
}

; CHECK-LABEL: define internal fastcc void @diam_bot(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @diam_bot(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

; CHECK-LABEL: define internal fastcc void @diam_l(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @diam_l(ptr %p) {
  call fastcc void @diam_bot(ptr %p)
  ret void
}

; CHECK-LABEL: define internal fastcc void @diam_r(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @diam_r(ptr %p) {
  call fastcc void @diam_bot(ptr %p)
  ret void
}

; CHECK-LABEL: define internal fastcc void @diam_top(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @diam_top(ptr %p) {
  call fastcc void @diam_l(ptr %p)
  call fastcc void @diam_r(ptr %p)
  ret void
}

define amdgpu_kernel void @k_diam(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_diam(
; CHECK: call fastcc void @diam_top(ptr inreg %p)
  call fastcc void @diam_top(ptr %p)
  ret void
}

; An argument that is already inreg must be left untouched (no double attribute,
; no crash).

; CHECK-LABEL: define internal fastcc void @callee_already(
; CHECK-SAME: ptr inreg %p
define internal fastcc void @callee_already(ptr inreg %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

define amdgpu_kernel void @k_already(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_already(
; CHECK: call fastcc void @callee_already(ptr inreg %p)
  call fastcc void @callee_already(ptr inreg %p)
  ret void
}

; Invoke call sites are not audited for inreg ABI consistency under exceptional
; control flow, so promotion is skipped.

; CHECK-LABEL: define internal fastcc void @callee_invoke(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_invoke(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

define amdgpu_kernel void @k_invoke(ptr %p) personality ptr null {
; CHECK-LABEL: define amdgpu_kernel void @k_invoke(
; CHECK: invoke fastcc void @callee_invoke(ptr %p)
  invoke fastcc void @callee_invoke(ptr %p) to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %tok = landingpad { ptr, i32 }
           cleanup
  ret void
}

; Indirect call through a bitcast of the function pointer: the callee is not a
; direct reference to @callee_bitcast, so the pass cannot prove all call sites.

; CHECK-LABEL: define internal fastcc void @callee_bitcast(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_bitcast(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

define amdgpu_kernel void @k_bitcast(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_bitcast(
; CHECK: call void {{.*}}(ptr {{.*}}%p)
; CHECK-NOT: inreg
  %fn = bitcast ptr @callee_bitcast to ptr
  call void %fn(ptr %p)
  ret void
}

; A function stored into a global (non-call use) is not eligible even if there
; is also a direct call the pass can see.

; CHECK-LABEL: define internal fastcc void @callee_stored(
; CHECK-SAME: ptr %p
; CHECK-NOT: ptr inreg
define internal fastcc void @callee_stored(ptr %p) {
  store float 0.000000e+00, ptr %p
  ret void
}

@fn_slot = global ptr null

define amdgpu_kernel void @k_stored(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_stored(
; CHECK: call fastcc void @callee_stored(ptr %p)
  store ptr @callee_stored, ptr @fn_slot
  call fastcc void @callee_stored(ptr %p)
  ret void
}

; inlinehint and alwaysinline callees behave the same for inreg promotion when
; the callee remains an out-of-line call (the motivating Kokkos case).

; CHECK-LABEL: define internal fastcc void @callee_inlinehint_uniform(
; CHECK-SAME: ptr inreg {{.*}}%p
define internal fastcc void @callee_inlinehint_uniform(ptr %p, i32 %i) #0 {
  %g = getelementptr float, ptr %p, i32 %i
  %v = load float, ptr %g
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_inlinehint_uniform(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_inlinehint_uniform(
; CHECK: call fastcc void @callee_inlinehint_uniform(ptr inreg %p, i32 %tid)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  call fastcc void @callee_inlinehint_uniform(ptr %p, i32 %tid)
  ret void
}

; CHECK-LABEL: define internal fastcc void @callee_alwaysinline_uniform(
; CHECK-SAME: ptr inreg {{.*}}%p
define internal fastcc void @callee_alwaysinline_uniform(ptr %p, i32 %i) #1 {
  %g = getelementptr float, ptr %p, i32 %i
  %v = load float, ptr %g
  store float %v, ptr %p
  ret void
}

define amdgpu_kernel void @k_alwaysinline_uniform(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k_alwaysinline_uniform(
; CHECK: call fastcc void @callee_alwaysinline_uniform(ptr inreg %p, i32 %tid)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  call fastcc void @callee_alwaysinline_uniform(ptr %p, i32 %tid)
  ret void
}

attributes #0 = { inlinehint }
attributes #1 = { alwaysinline }

declare i32 @llvm.amdgcn.workitem.id.x()
