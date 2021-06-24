; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s

declare { i32, i32 } @llvm.amdgcn.ds.bvh.stack.rtn(i32, i32, <4 x i32>, i32 immarg)

define amdgpu_gs void @test_ds_bvh_stack(i32 %addr, i32 %data0, <4 x i32> %data1, i32 addrspace(1)* %out) {
; CHECK-LABEL: test_ds_bvh_stack:
; CHECK: ; %bb.0:
; CHECK-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT: s_waitcnt_vscnt null, 0x0
; CHECK-NEXT: ds_bvh_stack_rtn_b32 v1, v0, v1, v[2:5]
; CHECK-NEXT: s_waitcnt lgkmcnt(0)
; CHECK-NEXT: s_waitcnt_vscnt null, 0x0
; CHECK-NEXT: buffer_gl0_inv
; CHECK-NEXT: buffer_gl1_inv
; CHECK-NEXT: global_store_b32 v[6:7], v1, off
; CHECK-NEXT: s_endpgm
  %vdst = call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.rtn(i32 %addr, i32 %data0, <4 x i32> %data1, i32 0)
  %res = extractvalue { i32, i32 } %vdst, 0
  store i32 %res, i32 addrspace(1)* %out, align 4
  ret void
}

define amdgpu_gs void @test_ds_bvh_stack_1(i32 %addr, i32 %data0, <4 x i32> %data1, i32 addrspace(1)* %out) {
; CHECK-LABEL: test_ds_bvh_stack_1:
; CHECK: ; %bb.0:
; CHECK-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK-NEXT: s_waitcnt_vscnt null, 0x0
; CHECK-NEXT: ds_bvh_stack_rtn_b32 v1, v0, v1, v[2:5] offset:1
; CHECK-NEXT: s_waitcnt lgkmcnt(0)
; CHECK-NEXT: s_waitcnt_vscnt null, 0x0
; CHECK-NEXT: buffer_gl0_inv
; CHECK-NEXT: buffer_gl1_inv
; CHECK-NEXT: global_store_b32 v[6:7], v1, off
; CHECK-NEXT: s_endpgm
  %vdst = call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.rtn(i32 %addr, i32 %data0, <4 x i32> %data1, i32 1)
  %res = extractvalue { i32, i32 } %vdst, 0
  store i32 %res, i32 addrspace(1)* %out, align 4
  ret void
}
