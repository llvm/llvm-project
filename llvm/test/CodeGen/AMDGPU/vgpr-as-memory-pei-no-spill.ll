; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -verify-machineinstrs < %s -o /dev/null

; A large "VGPR as memory" file can reach up into the callee-saved VGPR range
; (v40-47, ...). The reserved file registers must never be callee-save spilled
; by PEI: saving one in the prologue and restoring it in the epilogue would undo
; a callee's write to the shared file. Here @dev writes a file register that
; lands in the callee-saved range (the file is relocated above v31 and spans
; into it), so check its body has no save/restore of that register.

@big = internal addrspace(13) global [48 x i32] poison

; CHECK-LABEL: dev:
; CHECK-NOT: v_accvgpr_write
; CHECK-NOT: v_accvgpr_read
; CHECK-NOT: scratch_store
; CHECK-NOT: scratch_load
; CHECK-NOT: buffer_store
; CHECK-NOT: buffer_load
; CHECK: s_setpc_b64
define void @dev(i32 %v) {
  store i32 %v, ptr addrspace(13) @big
  %p = getelementptr [48 x i32], ptr addrspace(13) @big, i32 0, i32 45
  store i32 %v, ptr addrspace(13) %p
  ret void
}

define amdgpu_kernel void @k(i32 %v) {
  call void @dev(i32 %v)
  ret void
}
