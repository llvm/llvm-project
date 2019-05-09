; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=VI -check-prefix=VI-OPT %s
; RUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=VI -check-prefix=VI-NOOPT %s

; VI-LABEL: {{^}}dpp_test:
; VI: v_mov_b32_e32 v0, s{{[0-9]+}}
; VI: v_mov_b32_e32 v1, s{{[0-9]+}}
; VI-OPT: s_nop 1
; VI-NOOPT: s_nop 0
; VI-NOOPT: s_nop 0
; VI: v_mov_b32_dpp v0, v1 quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x01,0x08,0x11]
define amdgpu_kernel void @dpp_test(i32 addrspace(1)* %out, i32 %in1, i32 %in2) {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 1) #0
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; VI-LABEL: {{^}}dpp_test1:
; VI-OPT: v_add_u32_e32 [[REG:v[0-9]+]], vcc, v{{[0-9]+}}, v{{[0-9]+}}
; VI-NOOPT: v_add_u32_e64 [[REG:v[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-NOOPT: v_mov_b32_e32 v{{[0-9]+}}, 0
; VI-NEXT: s_nop 0
; VI-NEXT: s_nop 0
; VI-NEXT: v_mov_b32_dpp {{v[0-9]+}}, [[REG]] quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
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
