; RUN: llc -global-isel=0 -amdgpu-load-store-vectorizer=0 -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX10 %s
; RUN: llc -global-isel=1 -amdgpu-load-store-vectorizer=0 -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX10 %s
; RUN: llc -global-isel=0 -amdgpu-load-store-vectorizer=0 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX11 %s
; RUN: llc -global-isel=1 -amdgpu-load-store-vectorizer=0 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX11 %s

declare i32 @llvm.amdgcn.permlane16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()

; GCN-LABEL: {{^}}v_permlane16_b32_vss:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vii:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2{{$}}
define amdgpu_kernel void @v_permlane16_b32_vii(ptr addrspace(1) %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 1, i32 2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vll:
; FIXME-GFX10PLUS: It is allowed to have both immediates as literals
; GFX10PLUS-DAG: s_movk_i32 [[SRC1:s[0-9]+]], 0x1234
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], 0xc1d1{{$}}
define amdgpu_kernel void @v_permlane16_b32_vll(ptr addrspace(1) %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 4660, i32 49617, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vvv:
; GFX10-DAG: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10-DAG: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v1
; GFX11-DAG: v_and_b32_e32 [[VSRC1:v[0-9]+]],
; GFX11-DAG: v_bfe_u32 [[VSRC2:v[0-9]+]],
; GFX11-DAG: v_readfirstlane_b32 [[SRC1:s[0-9]+]], [[VSRC1]]
; GFX11-DAG: v_readfirstlane_b32 [[SRC2:s[0-9]+]], [[VSRC2]]
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vvv(ptr addrspace(1) %out, i32 %src0) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %tidx, i32 %tidy, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vvs:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_vvs(ptr addrspace(1) %out, i32 %src0, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %tidx, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vsv:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v{{[0-9]+}}
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vsv(ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %tidy, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vss_fi:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,0|1,0,0,1}}]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vss_bc:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{0,1|0,1,0,0}}]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vss_fi_bc:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,1|1,1,0,1}}]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vii:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vii(ptr addrspace(1) %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 1, i32 2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vll:
; FIXME-GFX10PLUS: It is allowed to have both immediates as literals
; GFX10PLUS-DAG: s_movk_i32 [[SRC1:s[0-9]+]], 0x1234
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], 0xc1d1{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vll(ptr addrspace(1) %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 4660, i32 49617, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vvv:
; GFX10-DAG: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10-DAG: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v1
; GFX11-DAG: v_and_b32_e32 [[VSRC1:v[0-9]+]],
; GFX11-DAG: v_bfe_u32 [[VSRC2:v[0-9]+]],
; GFX11-DAG: v_readfirstlane_b32 [[SRC1:s[0-9]+]], [[VSRC1]]
; GFX11-DAG: v_readfirstlane_b32 [[SRC2:s[0-9]+]], [[VSRC2]]
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vvv(ptr addrspace(1) %out, i32 %src0) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %tidx, i32 %tidy, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vvs:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vvs(ptr addrspace(1) %out, i32 %src0, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %tidx, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vsv:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v{{[0-9]+}}
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vsv(ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %tidy, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss_fi:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,0|1,0,0,1}}]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss_bc:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{0,1|0,1,0,0}}]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss_fi_bc:
; GFX10PLUS-NOT: v_readfirstlane_b32
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,1|1,1,0,1}}]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_tid_tid:
; GFX10PLUS: v_permlane16_b32 v0, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_tid_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %tidx, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_undef_tid:
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_undef_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 undef, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid:
; GFX10PLUS: v_{{(dual_)?}}mov_b32{{(_e32)?}} [[OLD:v[0-9]+]], 0x3039
; GFX10PLUS: v_permlane16_b32 [[OLD]], v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid_fi:
; GFX10PLUS-NOT: 0x3039
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,0|1,0,0,1}}]{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid_bc:
; GFX10PLUS-NOT: 0x3039
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{0,1|0,1,0,0}}]{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid_fi_bc:
; GFX10PLUS-NOT: 0x3039
; GFX10PLUS: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,1|1,1,0,1}}]{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_tid_tid:
; GFX10PLUS: v_permlanex16_b32 v0, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_tid_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %tidx, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_undef_tid:
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_undef_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 undef, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid:
; GFX10PLUS: v_{{(dual_)?}}mov_b32{{(_e32)?}} [[OLD:v[0-9]+]], 0x3039
; GFX10PLUS: v_permlanex16_b32 [[OLD]], v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid_fi:
; GFX10PLUS-NOT: 0x3039
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,0|1,0,0,1}}]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid_bc:
; GFX10PLUS-NOT: 0x3039
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{0,1|0,1,0,0}}]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid_fi_bc:
; GFX10PLUS-NOT: 0x3039
; GFX10PLUS: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[{{1,1|1,1,0,1}}]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
