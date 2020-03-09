;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=SI
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=SI
;RUN: llc < %s -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=GFX10

;CHECK-LABEL: {{^}}raw_atomic_buffer_load
;CHECK-LABEL: BB0_1: ; %bb1
;CHECK-NEXT: ; =>This Inner Loop Header: Depth=1
;CHECK-NEXT: s_waitcnt lgkmcnt(0)
;CHECK-NEXT: buffer_load_dword v1, off, s[0:3], 0 offset:4 glc
;CHECK-NEXT: s_waitcnt vmcnt(0)
;SI-NEXT: v_cmp_ne_u32_e32 vcc, v1, v0
;GFX10-NEXT: v_cmp_ne_u32_e32 vcc_lo, v1, v0
;SI-NEXT: s_or_b64 s[4:5], vcc, s[4:5]
;GFX10-NEXT: s_or_b32 s4, vcc_lo, s4
;SI-NEXT: s_andn2_b64 exec, exec, s[4:5]
;GFX10-NEXT: s_andn2_b32 exec_lo, exec_lo, s4
;CHECK-NEXT: s_cbranch_execnz BB0_1
define amdgpu_kernel void @raw_atomic_buffer_load(<4 x i32> %addr) {
bb:
  %tmp0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1
bb1:
  %0 = call i32 @llvm.amdgcn.raw.atomic.buffer.load.i32(<4 x i32> %addr, i32 4, i32 0, i32 1)
  %1 = icmp eq i32 %0, %tmp0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret void
}

;CHECK-LABEL: {{^}}raw_nonatomic_buffer_load
;CHECK: ; =>This Inner Loop Header: Depth=1
;SI-NEXT: s_and_b64 s[2:3], exec, vcc
;GFX10-NEXT: s_and_b32 s1, exec_lo, vcc_lo
;SI-NEXT: s_or_b64 s[0:1], s[2:3], s[0:1]
;GFX10-NEXT: s_or_b32 s0, s1, s0
;SI-NEXT: s_andn2_b64 exec, exec, s[0:1]
;GFX10-NEXT: s_andn2_b32 exec_lo, exec_lo, s0
;CHECK-NEXT: s_cbranch_execnz BB1_1
define amdgpu_kernel void @raw_nonatomic_buffer_load(<4 x i32> %addr) {
bb:
  %tmp0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1
bb1:
  %0 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %addr, i32 4, i32 0, i32 1)
  %1 = icmp eq i32 %0, %tmp0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @llvm.amdgcn.raw.atomic.buffer.load.i32(<4 x i32>, i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32 immarg) 
declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32 immarg) 
declare i32 @llvm.amdgcn.workitem.id.x()

