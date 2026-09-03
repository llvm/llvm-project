; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; An object in the VGPR "as memory" address space (13) occupies whole registers
; named by its address, and nothing may move or spill it, so it has to fit in
; the registers the function is entitled to. That budget is the one promote
; alloca already computes for vectorization: for a non-entry function it is the
; 32 registers the ABI preserves, unless the function is known to be inlined.
;
; Without this the overflow only surfaces from register allocation, as "ran out
; of registers" from a pass that has no idea what this address space is - and
; only once the object is large enough to starve everything else, so an object
; that merely overruns the budget is accepted and the function quietly claims
; more registers than it may use.

declare void @llvm.lifetime.start.p13(ptr addrspace(13) nocapture)
declare void @llvm.lifetime.end.p13(ptr addrspace(13) nocapture)

; 33 registers in a non-entry function, which has 32.
; CHECK: error: {{.*}}in function too_big_for_nonentry{{.*}}object in the VGPR 'as memory' address space (13) at v[0:32] does not fit in the 32 registers available to this function
define i32 @too_big_for_nonentry(i32 inreg %i) {
  %p = alloca [33 x float], align 4, addrspace(13)
  %q = getelementptr i32, ptr addrspace(13) %p, i32 %i
  %x = load i32, ptr addrspace(13) %q
  ret i32 %x
}

; The registers are counted across objects, not per object: two that each fit
; on their own do not both fit.
; CHECK: error: {{.*}}in function two_objects_overflow{{.*}}at v[16:32] does not fit in the 32 registers
define i32 @two_objects_overflow(i32 inreg %i) {
  %a = alloca [16 x float], align 4, addrspace(13)
  %b = alloca [17 x float], align 4, addrspace(13)
  %qa = getelementptr i32, ptr addrspace(13) %a, i32 %i
  %qb = getelementptr i32, ptr addrspace(13) %b, i32 %i
  %xa = load i32, ptr addrspace(13) %qa
  %xb = load i32, ptr addrspace(13) %qb
  %s = add i32 %xa, %xb
  ret i32 %s
}

; An address given by hand is checked the same way, since the budget is a
; property of the function rather than of how the object was placed.
; CHECK: error: {{.*}}in function preplaced_out_of_range{{.*}}at v[60:63] does not fit in the 32 registers
define i32 @preplaced_out_of_range(i32 inreg %i) {
  %p = alloca [4 x float], align 4, addrspace(13), !amdgpu.allocated.vgprs !0
  %q = getelementptr i32, ptr addrspace(13) %p, i32 %i
  %x = load i32, ptr addrspace(13) %q
  ret i32 %x
}

; Exactly the budget is not an overflow.
; CHECK-NOT: in function exactly_the_budget
define i32 @exactly_the_budget(i32 inreg %i) {
  %p = alloca [32 x float], align 4, addrspace(13)
  %q = getelementptr i32, ptr addrspace(13) %p, i32 %i
  %x = load i32, ptr addrspace(13) %q
  ret i32 %x
}

; A function that will be inlined is not held to the non-entry budget, which is
; how a caller-sized object reaches a helper.
; CHECK-NOT: in function inlined_helper
define i32 @inlined_helper(i32 inreg %i) alwaysinline {
  %p = alloca [33 x float], align 4, addrspace(13)
  %q = getelementptr i32, ptr addrspace(13) %p, i32 %i
  %x = load i32, ptr addrspace(13) %q
  ret i32 %x
}

; An entry function keeps its own, larger budget.
; CHECK-NOT: in function entry_function
define amdgpu_kernel void @entry_function(ptr addrspace(1) %out, i32 %i) {
  %p = alloca [33 x float], align 4, addrspace(13)
  %q = getelementptr i32, ptr addrspace(13) %p, i32 %i
  %x = load i32, ptr addrspace(13) %q
  store i32 %x, ptr addrspace(1) %out
  ret void
}

; v[60:63], well past the 32 a non-entry function may use.
!0 = !{i32 240, i32 16}
