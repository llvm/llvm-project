; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; The first load produces address in a VGPR which is used in address calculation
; of the second load (one inside the loop). The value is uniform and the inner
; load correctly selected to use SADDR form, however the address is promoted to
; vector registers because it all starts with a VGPR produced by the entry block
; load.
;
; Check that we are changing SADDR form of a load to VADDR and do not have to use
; readfirstlane instructions to move address from VGPRs into SGPRs.

; GCN-LABEL: {{^}}test_move_load_address_to_vgpr:
; GCN: BB{{[0-9]+}}_1:
; GCN-NOT: v_readfirstlane_b32
; GCN: global_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}], off glc
define amdgpu_kernel void @test_move_load_address_to_vgpr(i32 addrspace(1)* nocapture %arg) {
bb:
  %i1 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 0
  %i2 = load volatile i32, i32 addrspace(1)* %i1, align 4
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %i = phi i32 [ %i2, %bb ], [ %i8, %bb3 ]
  %i4 = zext i32 %i to i64
  %i5 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %i4
  %i6 = load volatile i32, i32 addrspace(1)* %i5, align 4
  %i8 = add nuw nsw i32 %i, 1
  %i9 = icmp eq i32 %i8, 256
  br i1 %i9, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}test_move_load_address_to_vgpr_d16_hi:
; GCN-NOT: v_readfirstlane_b32
; GCN: global_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}], off glc
define amdgpu_kernel void @test_move_load_address_to_vgpr_d16_hi(i16 addrspace(1)* nocapture %arg) {
bb:
  %i1 = getelementptr inbounds i16, i16 addrspace(1)* %arg, i64 0
  %load.pre = load volatile i16, i16 addrspace(1)* %i1, align 4
  %i2 = zext i16 %load.pre to i32
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %i = phi i32 [ %i2, %bb ], [ %i8, %bb3 ]
  %i4 = zext i32 %i to i64
  %i5 = getelementptr inbounds i16, i16 addrspace(1)* %arg, i64 %i4
  %i6 = load volatile i16, i16 addrspace(1)* %i5, align 4
  %insertelt = insertelement <2 x i16> undef, i16 %i6, i32 1
  %i8 =  bitcast <2 x i16> %insertelt to i32
  %i9 = icmp eq i32 %i8, 256
  br i1 %i9, label %bb2, label %bb3
}
