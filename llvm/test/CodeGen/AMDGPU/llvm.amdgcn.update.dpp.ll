; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}dpp_test:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @dpp_test(i32 addrspace(1)* %out, i32 %in1, i32 %in2) {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 0) #0
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_bc:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[2,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0{{$}}
define amdgpu_kernel void @dpp_test_bc(i32 addrspace(1)* %out, i32 %in1, i32 %in2) {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 2, i32 1, i32 1, i1 1) #0
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}


; VI-LABEL: {{^}}dpp_test1:
; GFX10: v_add_nc_u32_e32 [[REG:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX8-OPT: v_add_u32_e32 [[REG:v[0-9]+]], vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX8-NOOPT: v_add_u32_e64 [[REG:v[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX8-NOOPT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GFX8: s_nop 0
; GFX8-NEXT: s_nop 0
; GFX8-OPT-NEXT: v_mov_b32_dpp {{v[0-9]+}}, [[REG]] quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
@0 = internal unnamed_addr addrspace(3) global [448 x i32] undef, align 4
define weak_odr amdgpu_kernel void @dpp_test1(i32* %arg) local_unnamed_addr {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds [448 x i32], [448 x i32] addrspace(3)* @0, i32 0, i32 %tmp
  %tmp3 = load i32, i32 addrspace(3)* %tmp2, align 4
  fence syncscope("workgroup-one-as") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup-one-as") acquire
  %tmp4 = add nsw i32 %tmp3, %tmp3
  %tmp5 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp4, i32 177, i32 15, i32 15, i1 zeroext false)
  %tmp6 = add nsw i32 %tmp5, %tmp4
  %tmp7 = getelementptr inbounds i32, i32* %arg, i64 %tmp1
  store i32 %tmp6, i32* %tmp7, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0

attributes #0 = { nounwind readnone convergent }
