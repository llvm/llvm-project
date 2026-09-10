; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; An object in the VGPR "as memory" address space (13) occupies the registers
; its address names, and the liveness the backend builds keeps register
; allocation off them. Inline asm is not bound by that: it names physical
; registers directly. So an object live across asm that clobbers the registers
; it occupies would quietly read back whatever the asm left, and is diagnosed.
;
; Unlike a call, which is rejected outright because a callee may write any
; caller-saved register, asm is only rejected when it actually names registers
; the object occupies.

declare void @llvm.lifetime.end.p13(ptr addrspace(13) nocapture)

; The object is placed at v[0:7], which is exactly what the asm clobbers. The
; diagnostic names the registers it occupies, since that is what the asm's
; clobber list has to be compared against.
; CHECK: error: {{.*}}in function asm_clobbers_object{{.*}}object in the VGPR 'as memory' address space (13) at v[0:7] is clobbered by inline asm
define amdgpu_kernel void @asm_clobbers_object(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [8 x i32], addrspace(13)
  %p = getelementptr [8 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void asm sideeffect "; clobber", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"()
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; The same asm, with the object placed elsewhere: giving it an address that puts
; it at v[40:47] resolves the conflict, which is one of the two ways out of the
; diagnostic above.
; CHECK-NOT: in function asm_object_moved_away
define amdgpu_kernel void @asm_object_moved_away(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [8 x i32], addrspace(13), !amdgpu.allocated.vgprs !0
  %p = getelementptr [8 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void asm sideeffect "; clobber", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"()
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

!0 = !{i32 160, i32 32}

; Registers the object does not occupy are none of its business.
; CHECK-NOT: in function asm_unrelated_vgpr
define amdgpu_kernel void @asm_unrelated_vgpr(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [8 x i32], addrspace(13)
  %p = getelementptr [8 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void asm sideeffect "; nothing", "~{v100},~{v101}"()
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; The object never lives in scalar registers.
; CHECK-NOT: in function asm_sgpr_only
define amdgpu_kernel void @asm_sgpr_only(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [8 x i32], addrspace(13)
  %p = getelementptr [8 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void asm sideeffect "; nothing", "~{s40},~{s41}"()
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Asm that clobbers nothing cannot disturb the object.
; CHECK-NOT: in function asm_no_clobber
define amdgpu_kernel void @asm_no_clobber(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [8 x i32], addrspace(13)
  %p = getelementptr [8 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void asm sideeffect "s_nop 0", ""()
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; The lifetime ends first, so the registers are free by the time the asm runs.
; CHECK-NOT: in function asm_after_lifetime
define amdgpu_kernel void @asm_after_lifetime(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [8 x i32], addrspace(13)
  %p = getelementptr [8 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  call void @llvm.lifetime.end.p13(ptr addrspace(13) %obj)
  call void asm sideeffect "; clobber", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"()
  ret void
}
