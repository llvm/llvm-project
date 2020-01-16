; RUN: llc -amdgpu-load-store-vectorizer=0 -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

declare i32 @llvm.amdgcn.permlane16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()

; GCN-LABEL: {{^}}v_permlane16_b32_vss:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vii:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2{{$}}
define amdgpu_kernel void @v_permlane16_b32_vii(i32 addrspace(1)* %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 1, i32 2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vll:
; FIXME-GFX10: It is allowed to have both immediates as literals
; GFX10-DAG: s_movk_i32 [[SRC1:s[0-9]+]], 0x1234
; GFX10-DAG: s_mov_b32 [[SRC2:s[0-9]+]], 0xc1d1
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vll(i32 addrspace(1)* %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 4660, i32 49617, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vvv:
; GFX10-DAG: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10-DAG: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v1
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vvv(i32 addrspace(1)* %out, i32 %src0) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %tidx, i32 %tidy, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vvs:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_vvs(i32 addrspace(1)* %out, i32 %src0, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %tidx, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vsv:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v1
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vsv(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #1 {
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %tidy, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vss_fi:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,0]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss_fi(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vss_bc:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_vss_fi_bc:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,1]{{$}}
define amdgpu_kernel void @v_permlane16_b32_vss_fi_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vii:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vii(i32 addrspace(1)* %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 1, i32 2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vll:
; FIXME-GFX10: It is allowed to have both immediates as literals
; GFX10-DAG: s_movk_i32 [[SRC1:s[0-9]+]], 0x1234
; GFX10-DAG: s_mov_b32 [[SRC2:s[0-9]+]], 0xc1d1
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vll(i32 addrspace(1)* %out, i32 %src0) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 4660, i32 49617, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vvv:
; GFX10-DAG: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10-DAG: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v1
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vvv(i32 addrspace(1)* %out, i32 %src0) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %tidx, i32 %tidy, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vvs:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_readfirstlane_b32 [[SRC1:s[0-9]+]], v0
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[SRC1]], s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vvs(i32 addrspace(1)* %out, i32 %src0, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %tidx, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vsv:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_readfirstlane_b32 [[SRC2:s[0-9]+]], v1
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, [[SRC2]]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vsv(i32 addrspace(1)* %out, i32 %src0, i32 %src1) #1 {
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %tidy, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss_fi:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,0]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss_fi(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss_bc:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_vss_fi_bc:
; GFX10-NOT: v_readfirstlane_b32
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,1]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_vss_fi_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_tid_tid:
; GFX10: v_permlane16_b32 v0, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_tid_tid(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %tidx, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_undef_tid:
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_undef_tid(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 undef, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid:
; GFX10: v_mov_b32_e32 [[OLD:v[0-9]+]], 0x3039
; GFX10: v_permlane16_b32 [[OLD]], v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid_fi:
; GFX10-NOT: 0x3039
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,0]{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid_fi(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid_bc:
; GFX10-NOT: 0x3039
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1]{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlane16_b32_i_tid_fi_bc:
; GFX10-NOT: 0x3039
; GFX10: v_permlane16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,1]{{$}}
define amdgpu_kernel void @v_permlane16_b32_i_tid_fi_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_tid_tid:
; GFX10: v_permlanex16_b32 v0, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_tid_tid(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %tidx, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_undef_tid:
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_undef_tid(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 undef, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid:
; GFX10: v_mov_b32_e32 [[OLD:v[0-9]+]], 0x3039
; GFX10: v_permlanex16_b32 [[OLD]], v0, s{{[0-9]+}}, s{{[0-9]+}}{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid_fi:
; GFX10-NOT: 0x3039
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,0]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid_fi(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid_bc:
; GFX10-NOT: 0x3039
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 0, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_permlanex16_b32_i_tid_fi_bc:
; GFX10-NOT: 0x3039
; GFX10: v_permlanex16_b32 v{{[0-9]+}}, v0, s{{[0-9]+}}, s{{[0-9]+}} op_sel:[1,1]{{$}}
define amdgpu_kernel void @v_permlanex16_b32_i_tid_fi_bc(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 1, i1 1)
  store i32 %v, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
