; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs=0 < %s | FileCheck %s
; FIXME: These tests cannot be tail called, and should be executed in a waterfall loop.

declare hidden void @void_func_i32_inreg(i32 inreg)
define void @tail_call_i32_inreg_divergent(i32 %vgpr) {
  tail call void @void_func_i32_inreg(i32 inreg %vgpr)
  ret void
}

@constant = external hidden addrspace(4) constant ptr

define void @indirect_tail_call_i32_inreg_divergent(i32 %vgpr) {
  %fptr = load ptr, ptr addrspace(4) @constant, align 8
  tail call void %fptr(i32 inreg %vgpr)
  ret void
}
;CHECK: buffer_store_dword v40, off, s[0:3], s33 ; 4-byte Folded Spill
;CHECK: s_mov_b64 exec, s[18:19]
;CHECK:	v_writelane_b32 v40, s16, 2
;CHECK:	s_addk_i32 s32, 0x400
;CHECK:	v_writelane_b32 v40, s30, 0
;CHECK:	s_getpc_b64 s[16:17]
;CHECK:	s_add_u32 s16, s16, void_func_i32_inreg@rel32@lo+4
;CHECK:	s_addc_u32 s17, s17, void_func_i32_inreg@rel32@hi+12
;CHECK:	v_readfirstlane_b32 s0, v0
;CHECK:	v_writelane_b32 v40, s31, 1
;CHECK:	s_swappc_b64 s[30:31], s[16:17]
;CHECK:	v_readlane_b32 s31, v40, 1
;CHECK:	v_readlane_b32 s30, v40, 0
;CHECK:	s_mov_b32 s32, s33
;CHECK:	v_readlane_b32 s4, v40, 2
;CHECK:	s_or_saveexec_b64 s[6:7], -1
;CHECK:	buffer_load_dword v40, off, s[0:3], s33