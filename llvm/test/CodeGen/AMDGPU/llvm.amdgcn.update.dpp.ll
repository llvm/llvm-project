; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX8-OPT,GCN-OPT %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -O0 -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX8-NOOPT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX10,GCN-OPT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=-flat-for-global -amdgpu-enable-vopd=0 -amdgpu-dpp-combine=false -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX11,GCN-OPT %s

; GCN-LABEL: {{^}}dpp_test:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @dpp_test(ptr addrspace(1) %out, i32 %in1, i32 %in2) {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 false) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_bc:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[2,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_bc(ptr addrspace(1) %out, i32 %in1, i32 %in2) {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 2, i32 1, i32 1, i1 true) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}


; GCN-LABEL: {{^}}dpp_test1:
; GFX10,GFX11: v_add_nc_u32_e32 [[REG:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX8-OPT: v_add_u32_e32 [[REG:v[0-9]+]], vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX8-NOOPT: v_add_u32_e64 [[REG:v[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX8-NOOPT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GFX8: s_nop 1
; GFX8-NEXT: v_mov_b32_dpp {{v[0-9]+}}, [[REG]] quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
@0 = internal unnamed_addr addrspace(3) global [448 x i32] undef, align 4
define weak_odr amdgpu_kernel void @dpp_test1(ptr %arg) local_unnamed_addr {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds [448 x i32], ptr addrspace(3) @0, i32 0, i32 %tmp
  %tmp3 = load i32, ptr addrspace(3) %tmp2, align 4
  fence syncscope("workgroup-one-as") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup-one-as") acquire
  %tmp4 = add nsw i32 %tmp3, %tmp3
  %tmp5 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %tmp4, i32 177, i32 15, i32 15, i1 zeroext false)
  %tmp6 = add nsw i32 %tmp5, %tmp4
  %tmp7 = getelementptr inbounds i32, ptr %arg, i64 %tmp1
  store i32 %tmp6, ptr %tmp7, align 4
  ret void
}

; GCN-LABEL: {{^}}update_dppi64_test:
; GCN:     load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppi64_test(ptr addrspace(1) %arg, i64 %in1, i64 %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load i64, ptr addrspace(1) %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 1, i32 1, i32 1, i1 false) #0
  store i64 %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dppf64_test:
; GCN:     load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppf64_test(ptr addrspace(1) %arg, double %in1, double %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, ptr addrspace(1) %arg, i32 %id
  %load = load double, ptr addrspace(1) %gep
  %tmp0 = call double @llvm.amdgcn.update.dpp.f64(double %in1, double %load, i32 1, i32 1, i32 1, i1 false) #0
  store double %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dppv2i32_test:
; GCN:     load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppv2i32_test(ptr addrspace(1) %arg, <2 x i32> %in1, <2 x i32> %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x i32>, ptr addrspace(1) %arg, i32 %id
  %load = load <2 x i32>, ptr addrspace(1) %gep
  %tmp0 = call <2 x i32> @llvm.amdgcn.update.dpp.v2i32(<2 x i32> %in1, <2 x i32> %load, i32 1, i32 1, i32 1, i1 false) #0
  store <2 x i32> %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dppv2f32_test:
; GCN:     load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppv2f32_test(ptr addrspace(1) %arg, <2 x float> %in1, <2 x float> %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %arg, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep
  %tmp0 = call <2 x float> @llvm.amdgcn.update.dpp.v2f32(<2 x float> %in1, <2 x float> %load, i32 1, i32 1, i32 1, i1 false) #0
  store <2 x float> %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dpp_p0_test:
; GCN:     load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dpp_p0_test(ptr addrspace(1) %arg, ptr %in1, ptr %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds ptr, ptr addrspace(1) %arg, i32 %id
  %load = load ptr, ptr addrspace(1) %gep
  %tmp0 = call ptr @llvm.amdgcn.update.dpp.p0(ptr %in1, ptr %load, i32 1, i32 1, i32 1, i1 false) #0
  store ptr %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dpp_p3_test:
; GCN: {{load|read}}_{{dword|b32}} v[[SRC:[0-9]+]]
; GCN: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dpp_p3_test(ptr addrspace(3) %arg, ptr addrspace(3) %in1, ptr %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds ptr addrspace(3), ptr addrspace(3) %arg, i32 %id
  %load = load ptr addrspace(3), ptr addrspace(3) %gep
  %tmp0 = call ptr addrspace(3) @llvm.amdgcn.update.dpp.p3(ptr addrspace(3) %in1, ptr addrspace(3) %load, i32 1, i32 1, i32 1, i1 false) #0
  store ptr addrspace(3) %tmp0, ptr addrspace(3) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dpp_p5_test:
; GCN: {{load|read}}_{{dword|b32}} v[[SRC:[0-9]+]]
; GCN: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dpp_p5_test(ptr addrspace(5) %arg, ptr addrspace(5) %in1, ptr %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds ptr addrspace(5), ptr addrspace(5) %arg, i32 %id
  %load = load ptr addrspace(5), ptr addrspace(5) %gep
  %tmp0 = call ptr addrspace(5) @llvm.amdgcn.update.dpp.p5(ptr addrspace(5) %in1, ptr addrspace(5) %load, i32 1, i32 1, i32 1, i1 false) #0
  store ptr addrspace(5) %tmp0, ptr addrspace(5) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dppi64_imm_old_test:
; GCN-OPT-DAG: v_mov_b32_e32 v[[OLD_LO:[0-9]+]], 0x3afaedd9
; GFX8-OPT-DAG,GFX10-DAG: v_mov_b32_e32 v[[OLD_HI:[0-9]+]], 0x7047
; GFX11-DAG: v_mov_b32_e32 v[[OLD_HI:[0-9]+]], 0x7047
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_LO:[0-9]+]], 0x3afaedd9
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_HI:[0-9]+]], 0x7047
; GCN-DAG: load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-OPT-DAG: v_mov_b32_dpp v[[OLD_LO]], v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GFX8-OPT-DAG,GFX10-DAG,GFX11-DAG: v_mov_b32_dpp v[[OLD_HI]], v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppi64_imm_old_test(ptr addrspace(1) %arg, i64 %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load i64, ptr addrspace(1) %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 123451234512345, i64 %load, i32 1, i32 1, i32 1, i1 false) #0
  store i64 %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dppf64_imm_old_test:
; GCN-OPT-DAG: v_mov_b32_e32 v[[OLD_LO:[0-9]+]], 0x6b8564a
; GFX8-OPT-DAG,GFX10-DAG: v_mov_b32_e32 v[[OLD_HI:[0-9]+]], 0x405edce1
; GFX11-DAG: v_mov_b32_e32 v[[OLD_HI:[0-9]+]], 0x405edce1
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_LO:[0-9]+]], 0x6b8564a
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_HI:[0-9]+]], 0x405edce1
; GCN-DAG: load_{{dwordx2|b64}} v[[[SRC_LO:[0-9]+]]:[[SRC_HI:[0-9]+]]]
; GCN-OPT-DAG: v_mov_b32_dpp v[[OLD_LO]], v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GFX8-OPT-DAG,GFX10-DAG,GFX11-DAG: v_mov_b32_dpp v[[OLD_HI]], v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppf64_imm_old_test(ptr addrspace(1) %arg, double %in2) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load double, ptr addrspace(1) %gep
  %tmp0 = call double @llvm.amdgcn.update.dpp.f64(double 123.4512345123450, double %load, i32 1, i32 1, i32 1, i1 false) #0
  store double %tmp0, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}update_dppi64_imm_src_test:
; GCN-OPT-DAG: v_mov_b32_e32 v[[OLD_LO:[0-9]+]], 0x3afaedd9
; GCN-OPT-DAG: v_mov_b32_e32 v[[OLD_HI:[0-9]+]], 0x7047
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_LO:[0-9]+]], 0x3afaedd9
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_HI:[0-9]+]], 0x7047
; GCN-OPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[OLD_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-OPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[OLD_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppi64_imm_src_test(ptr addrspace(1) %out, i64 %in1) {
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 123451234512345, i32 1, i32 1, i32 1, i1 false) #0
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}update_dppf64_imm_src_test:
; GCN-OPT-DAG: v_mov_b32_e32 v[[OLD_LO:[0-9]+]], 0x6b8564a
; GCN-OPT-DAG: v_mov_b32_e32 v[[OLD_HI:[0-9]+]], 0x405edce1
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_LO:[0-9]+]], 0x6b8564a
; GFX8-NOOPT-DAG: s_mov_b32 s[[SOLD_HI:[0-9]+]], 0x405edce1
; GCN-OPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[OLD_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-OPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[OLD_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_LO]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
; GCN-NOOPT-DAG: v_mov_b32_dpp v{{[0-9]+}}, v[[SRC_HI]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @update_dppf64_imm_src_test(ptr addrspace(1) %out, double %in1) {
  %tmp0 = call double @llvm.amdgcn.update.dpp.f64(double %in1, double 123.451234512345, i32 1, i32 1, i32 1, i1 false) #0
  store double %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @dpp_test_f32(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 1, i32 1, i32 1, i1 false)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb1:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,0,0,0] row_mask:0x0 bank_mask:0x0{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb1(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 0, i32 0, i32 0, i1 false)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb2:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,0,0,0] row_mask:0x3 bank_mask:0x3{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb2(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 3, i32 3, i32 3, i1 false)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb3:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x2 bank_mask:0x3 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb3(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 1, i32 2, i32 3, i1 true)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb4:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,1,0,0] row_mask:0x3 bank_mask:0x2 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb4(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 4, i32 3, i32 2, i1 true)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb5:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,3,0] row_mask:0xe bank_mask:0xd bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb5(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 63, i32 62, i32 61, i1 true)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb6:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,3,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb6(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 63, i32 63, i32 63, i1 true)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}


; GCN-LABEL: {{^}}dpp_test_f32_imm_comb7:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,0,0,1] row_mask:0x0 bank_mask:0x0 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb7(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 64, i32 64, i32 64, i1 true)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_f32_imm_comb8:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,1,0] row_mask:0xf bank_mask:0x0 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_f32_imm_comb8(ptr addrspace(1) %out, float %in1, float %in2) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %in1, float %in2, i32 31, i32 63, i32 128, i1 true)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @dpp_test_v2i16(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 1, i32 1, i32 1, i1 false)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb1:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,0,0,0] row_mask:0x0 bank_mask:0x0{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb1(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 0, i32 0, i32 0, i1 false)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb2:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,0,0,0] row_mask:0x3 bank_mask:0x3{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb2(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 3, i32 3, i32 3, i1 false)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

	; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb3:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x2 bank_mask:0x3 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb3(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 1, i32 2, i32 3, i1 true)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb4:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,1,0,0] row_mask:0x3 bank_mask:0x2 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb4(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 4, i32 3, i32 2, i1 true)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb5:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,3,0] row_mask:0xe bank_mask:0xd bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb5(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 63, i32 62, i32 61, i1 true)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb6:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,3,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb6(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 63, i32 63, i32 63, i1 true)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb7:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,0,0,1] row_mask:0x0 bank_mask:0x0 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb7(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 64, i32 64, i32 64, i1 true)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2i16_imm_comb8:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,1,0] row_mask:0xf bank_mask:0x0 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2i16_imm_comb8(ptr addrspace(1) %out, <2 x i16> %in1, <2 x i16> %in2) {
  %tmp0 = call <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16> %in1, <2 x i16> %in2, i32 31, i32 63, i32 128, i1 true)
  store <2 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1{{$}}
define amdgpu_kernel void @dpp_test_v2f16(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 1, i32 1, i32 1, i1 false)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb1:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,0,0,0] row_mask:0x0 bank_mask:0x0{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb1(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 0, i32 0, i32 0, i1 false)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb2:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,0,0,0] row_mask:0x3 bank_mask:0x3{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb2(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 3, i32 3, i32 3, i1 false)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

	; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb3:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[1,0,0,0] row_mask:0x2 bank_mask:0x3 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb3(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 1, i32 2, i32 3, i1 true)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb4:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,1,0,0] row_mask:0x3 bank_mask:0x2 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb4(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 4, i32 3, i32 2, i1 true)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb5:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,3,0] row_mask:0xe bank_mask:0xd bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb5(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 63, i32 62, i32 61, i1 true)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb6:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,3,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb6(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 63, i32 63, i32 63, i1 true)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb7:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[0,0,0,1] row_mask:0x0 bank_mask:0x0 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb7(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 64, i32 64, i32 64, i1 true)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dpp_test_v2f16_imm_comb8:
; GCN:  v_mov_b32_e32 [[DST:v[0-9]+]], s{{[0-9]+}}
; GCN:  v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX8-OPT: s_mov
; GFX8-OPT: s_mov
; GFX8-NOOPT: s_nop 1
; GCN:  v_mov_b32_dpp [[DST]], [[SRC]] quad_perm:[3,3,1,0] row_mask:0xf bank_mask:0x0 bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_test_v2f16_imm_comb8(ptr addrspace(1) %out, <2 x half> %in1, <2 x half> %in2) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> %in1, <2 x half> %in2, i32 31, i32 63, i32 128, i1 true)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0
declare <2 x i16> @llvm.amdgcn.update.dpp.v2i16(<2 x i16>, <2 x i16>, i32, i32, i32, i1) #0
declare <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half>, <2 x half>, i32, i32, i32, i1) #0
declare float @llvm.amdgcn.update.dpp.f32(float, float, i32, i32, i32, i1) #0
declare i64 @llvm.amdgcn.update.dpp.i64(i64, i64, i32, i32, i32, i1) #0

attributes #0 = { nounwind readnone convergent }
