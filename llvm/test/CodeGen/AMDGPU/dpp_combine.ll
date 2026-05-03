; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s -check-prefixes=GCN,GFX9GFX10
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 < %s | FileCheck %s -check-prefixes=GCN,GFX9GFX10
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1150 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1150 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1251 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1251 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16

; GCN-LABEL: {{^}}dpp_add:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_add_{{(nc_)?}}u32_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_add(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %add = add i32 %tmp0, %load
  store i32 %add, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp_ceil:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_ceil_f32_dpp [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_ceil(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i32 %tmp0 to float
  %round = tail call float @llvm.ceil.f32(float %tmp1)
  %tmp2 = bitcast float %round to i32
  store i32 %tmp2, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp_fadd:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_add_f32_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_fadd(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i32 %tmp0 to float
  %t = bitcast i32 %load to float
  %add = fadd float %tmp1, %t
  %tmp2 = bitcast float %add to i32
  store i32 %tmp2, ptr addrspace(1) %gep
  ret void
}

; Fails to combine prior to gfx1251 because v_mul_lo_u32 has no e32 or dpp form.
; Fails to combine on gfx1251 because DPP control value is invalid for DP DPP and v_mul_lo_u32 is
; classified as DP DPP.
; GCN-LABEL: {{^}}dpp_mul:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_mov_b32_e32 [[V2:v[0-9]+]], [[V]]
; GCN: v_mov_b32_dpp [[V2]], [[V2]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GCN: v_mul_lo_u32 [[V]], [[V2]], [[V]]{{$}}
define amdgpu_kernel void @dpp_mul(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1)
  %mul = mul i32 %tmp0, %load
  store i32 %mul, ptr addrspace(1) %gep
  ret void
}

; It is not expected to see a sequence of v_mov_b32_dpp feeding into a 16 bit instruction
; GCN-LABEL: {{^}}dpp_fadd_f16:
; GFX9GFX10: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GFX9GFX10: v_add_f16_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX11-TRUE16: v_mov_b32_dpp {{v[0-9]+}}, {{v[0-9]+}} quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
; GFX11-TRUE16: v_add_f16_e32
; GFX11-FAKE16: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GFX11-FAKE16: v_add_f16_e64_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
define amdgpu_kernel void @dpp_fadd_f16(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp01 = trunc i32 %tmp0 to i16
  %tmp1 = bitcast i16 %tmp01 to half
  %tt = trunc i32 %load to i16
  %t = bitcast i16 %tt to half
  %add = fadd half %tmp1, %t
  %tmp2 = bitcast half %add to i16
  %tmp3 = zext i16 %tmp2 to i32
  store i32 %tmp3, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp_fmin_f32:
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
define nofpclass(nan) float @dpp_fmin_f32(float nofpclass(nan) %x) {
entry:
  %dpp.shr1 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %x, i32 273, i32 15, i32 15, i1 false)
  %min1 = tail call nnan float @llvm.minnum.f32(float %x, float %dpp.shr1)
  %dpp.shr2 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %min1, i32 274, i32 15, i32 15, i1 false)
  %min2 = tail call nnan float @llvm.minnum.f32(float %min1, float %dpp.shr2)
  %dpp.shr4 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %min2, i32 276, i32 15, i32 15, i1 false)
  %min3 = tail call nnan float @llvm.minnum.f32(float %min2, float %dpp.shr4)
  %dpp.shr8 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %min3, i32 280, i32 15, i32 15, i1 false)
  %min4 = tail call nnan float @llvm.minnum.f32(float %min3, float %dpp.shr8)
  ret float %min4
}

; GCN-LABEL: {{^}}dpp_fmax_f32:
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
define nofpclass(nan) float @dpp_fmax_f32(float nofpclass(nan) %x) #0 {
entry:
  %dpp.shr1 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %x, i32 273, i32 15, i32 15, i1 false)
  %max1 = tail call nnan float @llvm.maxnum.f32(float %x, float %dpp.shr1)
  %dpp.shr2 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %max1, i32 274, i32 15, i32 15, i1 false)
  %max2 = tail call nnan float @llvm.maxnum.f32(float %max1, float %dpp.shr2)
  %dpp.shr4 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %max2, i32 276, i32 15, i32 15, i1 false)
  %max3 = tail call nnan float @llvm.maxnum.f32(float %max2, float %dpp.shr4)
  %dpp.shr8 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %max3, i32 280, i32 15, i32 15, i1 false)
  %max4 = tail call nnan float @llvm.maxnum.f32(float %max3, float %dpp.shr8)
  ret float %max4
}

; GCN-LABEL: {{^}}dpp_fminimum_f32:
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_min{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
define nofpclass(nan) float @dpp_fminimum_f32(float nofpclass(nan) %x) {
entry:
  %dpp.shr1 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %x, i32 273, i32 15, i32 15, i1 false)
  %min1 = tail call nnan float @llvm.minimumnum.f32(float %x, float %dpp.shr1)
  %dpp.shr2 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %min1, i32 274, i32 15, i32 15, i1 false)
  %min2 = tail call nnan float @llvm.minimumnum.f32(float %min1, float %dpp.shr2)
  %dpp.shr4 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %min2, i32 276, i32 15, i32 15, i1 false)
  %min3 = tail call nnan float @llvm.minimumnum.f32(float %min2, float %dpp.shr4)
  %dpp.shr8 = tail call float @llvm.amdgcn.update.dpp.f32(float 0x7FF0000000000000, float %min3, i32 280, i32 15, i32 15, i1 false)
  %min4 = tail call nnan float @llvm.minimumnum.f32(float %min3, float %dpp.shr8)
  ret float %min4
}

; GCN-LABEL: {{^}}dpp_fmaximum_f32:
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GCN: v_max{{(_num)?}}_f32_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
define nofpclass(nan) float @dpp_fmaximum_f32(float nofpclass(nan) %x) #0 {
entry:
  %dpp.shr1 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %x, i32 273, i32 15, i32 15, i1 false)
  %max1 = tail call nnan float @llvm.maximumnum.f32(float %x, float %dpp.shr1)
  %dpp.shr2 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %max1, i32 274, i32 15, i32 15, i1 false)
  %max2 = tail call nnan float @llvm.maximumnum.f32(float %max1, float %dpp.shr2)
  %dpp.shr4 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %max2, i32 276, i32 15, i32 15, i1 false)
  %max3 = tail call nnan float @llvm.maximumnum.f32(float %max2, float %dpp.shr4)
  %dpp.shr8 = tail call float @llvm.amdgcn.update.dpp.f32(float 0xFFF0000000000000, float %max3, i32 280, i32 15, i32 15, i1 false)
  %max4 = tail call nnan float @llvm.maximumnum.f32(float %max3, float %dpp.shr8)
  ret float %max4
}

; GCN-LABEL: {{^}}dpp_fmin_f16:
; GFX9GFX10: v_min_f16_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GFX9GFX10: v_min_f16_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GFX9GFX10: v_min_f16_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GFX9GFX10: v_min_f16_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-TRUE16: v_mov_b32_dpp {{v[0-9]+}}, {{v[0-9]+}} row_shr:1 row_mask:0xf bank_mask:0xf
; GFX11-TRUE16: v_min{{(_num)?}}_f16_e32
; GFX11-FAKE16: v_min{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-FAKE16: v_min{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-FAKE16: v_min{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-FAKE16: v_min{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
define nofpclass(nan) half @dpp_fmin_f16(half nofpclass(nan) %x) {
entry:
  %dpp.shr1 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xH7C00, half %x, i32 273, i32 15, i32 15, i1 false)
  %min1 = tail call nnan half @llvm.minnum.f16(half %x, half %dpp.shr1)
  %dpp.shr2 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xH7C00, half %min1, i32 274, i32 15, i32 15, i1 false)
  %min2 = tail call nnan half @llvm.minnum.f16(half %min1, half %dpp.shr2)
  %dpp.shr4 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xH7C00, half %min2, i32 276, i32 15, i32 15, i1 false)
  %min3 = tail call nnan half @llvm.minnum.f16(half %min2, half %dpp.shr4)
  %dpp.shr8 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xH7C00, half %min3, i32 280, i32 15, i32 15, i1 false)
  %min4 = tail call nnan half @llvm.minnum.f16(half %min3, half %dpp.shr8)
  ret half %min4
}

; GCN-LABEL: {{^}}dpp_fmax_f16:
; GFX9GFX10: v_max_f16_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GFX9GFX10: v_max_f16_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GFX9GFX10: v_max_f16_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GFX9GFX10: v_max_f16_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-TRUE16: v_mov_b32_dpp {{v[0-9]+}}, {{v[0-9]+}} row_shr:1 row_mask:0xf bank_mask:0xf
; GFX11-TRUE16: v_max{{(_num)?}}_f16_e32
; GFX11-FAKE16: v_max{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-FAKE16: v_max{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-FAKE16: v_max{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf{{$}}
; GFX11-FAKE16: v_max{{(_num)?}}_f16_e64_dpp v0, v0, v0 row_shr:8 row_mask:0xf bank_mask:0xf{{$}}
define nofpclass(nan) half @dpp_fmax_f16(half nofpclass(nan) %x) #0 {
entry:
  %dpp.shr1 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xHFC00, half %x, i32 273, i32 15, i32 15, i1 false)
  %max1 = tail call nnan half @llvm.maxnum.f16(half %x, half %dpp.shr1)
  %dpp.shr2 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xHFC00, half %max1, i32 274, i32 15, i32 15, i1 false)
  %max2 = tail call nnan half @llvm.maxnum.f16(half %max1, half %dpp.shr2)
  %dpp.shr4 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xHFC00, half %max2, i32 276, i32 15, i32 15, i1 false)
  %max3 = tail call nnan half @llvm.maxnum.f16(half %max2, half %dpp.shr4)
  %dpp.shr8 = tail call half @llvm.amdgcn.update.dpp.f16(half 0xHFC00, half %max3, i32 280, i32 15, i32 15, i1 false)
  %max4 = tail call nnan half @llvm.maxnum.f16(half %max3, half %dpp.shr8)
  ret half %max4
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0
declare float @llvm.ceil.f32(float)

attributes #0 = { nounwind readnone convergent }
