; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s

%struct.ByValStruct = type { [4 x i32] }

; GCN-LABEL: {{^}}void_func_byval_struct:
; GCN: s_mov_b32 s5, s32
; GCN: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-NOT: s32
; GCN: buffer_store_dword [[LOAD0]], off, s[0:3], s5 offset:4{{$}}
; GCN-NOT: s32

; GCN: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s5 offset:20{{$}}
; GCN-NOT: s32
; GCN: buffer_store_dword [[LOAD1]], off, s[0:3], s5 offset:20{{$}}
; GCN-NOT: s32
define void @void_func_byval_struct(%struct.ByValStruct addrspace(5)* byval noalias nocapture align 4 %arg0, %struct.ByValStruct addrspace(5)* byval noalias nocapture align 4 %arg1) #1 {
entry:
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  %tmp = load volatile i32, i32 addrspace(5)* %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  store volatile i32 %add, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  %tmp1 = load volatile i32, i32 addrspace(5)* %arrayidx2, align 4
  %add3 = add nsw i32 %tmp1, 2
  store volatile i32 %add3, i32 addrspace(5)* %arrayidx2, align 4
  store volatile i32 9, i32 addrspace(1)* null, align 4
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_non_leaf:
; GCN: s_mov_b32 s5, s32
; GCN-DAG: buffer_store_dword v32
; GCN-DAG: buffer_store_dword v33
; GCN-NOT: v_writelane_b32 v{{[0-9]+}}, s32
; GCN-DAG: v_writelane_b32
; GCN-DAG: s_add_u32 s32, s32, 0xc00{{$}}
; GCN-DAG: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: v_add_{{[iu]}}32_e32 [[ADD0:v[0-9]+]], vcc, 1, [[LOAD0]]
; GCN-DAG: buffer_store_dword [[ADD0]], off, s[0:3], s5 offset:4{{$}}

; GCN-DAG: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s5 offset:20{{$}}
; GCN-DAG: v_add_{{[iu]}}32_e32 [[ADD1:v[0-9]+]], vcc, 2, [[LOAD1]]

; GCN: s_swappc_b64

; GCN: buffer_store_dword [[ADD1]], off, s[0:3], s5 offset:20{{$}}

; GCN: v_readlane_b32
; GCN-NOT: v_readlane_b32 s32
; GCN-DAG: buffer_load_dword v32,
; GCN-DAG: buffer_load_dword v33,
; GCN: s_sub_u32 s32, s32, 0xc00{{$}}
; GCN: s_setpc_b64
define void  @void_func_byval_struct_non_leaf(%struct.ByValStruct addrspace(5)* byval noalias nocapture align 4 %arg0, %struct.ByValStruct addrspace(5)* byval noalias nocapture align 4 %arg1) #1 {
entry:
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  %tmp = load volatile i32, i32 addrspace(5)* %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  store volatile i32 %add, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  %tmp1 = load volatile i32, i32 addrspace(5)* %arrayidx2, align 4
  %add3 = add nsw i32 %tmp1, 2
  call void @external_void_func_void()
  store volatile i32 %add3, i32 addrspace(5)* %arrayidx2, align 4
  store volatile i32 9, i32 addrspace(1)* null, align 4
  ret void
}

; GCN-LABEL: {{^}}call_void_func_byval_struct_func:
; GCN: s_mov_b32 s5, s32
; GCN-DAG: s_add_u32 s32, s32, 0xc00{{$}}
; GCN-DAG: v_writelane_b32

; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 [[THIRTEEN:v[0-9]+]], 13

; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s5 offset:8
; GCN-DAG: buffer_store_dword [[THIRTEEN]], off, s[0:3], s5 offset:24

; GCN: buffer_load_dword [[LOAD4:v[0-9]+]], off, s[0:3], s5 offset:24
; GCN: buffer_load_dword [[LOAD5:v[0-9]+]], off, s[0:3], s5 offset:28
; GCN: buffer_load_dword [[LOAD6:v[0-9]+]], off, s[0:3], s5 offset:32
; GCN: buffer_load_dword [[LOAD7:v[0-9]+]], off, s[0:3], s5 offset:36

; GCN-DAG: buffer_store_dword [[LOAD4]], off, s[0:3], s32 offset:20
; GCN-DAG: buffer_store_dword [[LOAD5]], off, s[0:3], s32 offset:24
; GCN-DAG: buffer_store_dword [[LOAD6]], off, s[0:3], s32 offset:28
; GCN-DAG: buffer_store_dword [[LOAD7]], off, s[0:3], s32 offset:32

; GCN-DAG: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s5 offset:8
; GCN-DAG: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s5 offset:12
; GCN-DAG: buffer_load_dword [[LOAD2:v[0-9]+]], off, s[0:3], s5 offset:16
; GCN-DAG: buffer_load_dword [[LOAD3:v[0-9]+]], off, s[0:3], s5 offset:20

; GCN-NOT: s_add_u32 s32, s32, 0x800

; GCN-DAG: buffer_store_dword [[LOAD0]], off, s[0:3], s32 offset:4{{$}}
; GCN-DAG: buffer_store_dword [[LOAD1]], off, s[0:3], s32 offset:8
; GCN-DAG: buffer_store_dword [[LOAD2]], off, s[0:3], s32 offset:12
; GCN-DAG: buffer_store_dword [[LOAD3]], off, s[0:3], s32 offset:16


; GCN: s_swappc_b64
; GCN-NOT: v_readlane_b32 s32
; GCN: v_readlane_b32
; GCN-NOT: v_readlane_b32 s32

; GCN-NOT: s_sub_u32 s32, s32, 0x800

; GCN: s_sub_u32 s32, s32, 0xc00{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @call_void_func_byval_struct_func() #1 {
entry:
  %arg0 = alloca %struct.ByValStruct, align 4, addrspace(5)
  %arg1 = alloca %struct.ByValStruct, align 4, addrspace(5)
  %tmp = bitcast %struct.ByValStruct addrspace(5)* %arg0 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp)
  %tmp1 = bitcast %struct.ByValStruct addrspace(5)* %arg1 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  store volatile i32 9, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  store volatile i32 13, i32 addrspace(5)* %arrayidx2, align 4
  call void @void_func_byval_struct(%struct.ByValStruct addrspace(5)* byval nonnull align 4 %arg0, %struct.ByValStruct addrspace(5)* byval nonnull align 4 %arg1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp)
  ret void
}

; GCN-LABEL: {{^}}call_void_func_byval_struct_kernel:
; GCN: s_mov_b32 s33, s7
; GCN: s_add_u32 s32, s33, 0xc00{{$}}

; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 [[THIRTEEN:v[0-9]+]], 13
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s33 offset:8
; GCN: buffer_store_dword [[THIRTEEN]], off, s[0:3], s33 offset:24

; GCN-NOT: s_add_u32 s32, s32, 0x800

; GCN-DAG: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s33 offset:8
; GCN-DAG: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s33 offset:12
; GCN-DAG: buffer_load_dword [[LOAD2:v[0-9]+]], off, s[0:3], s33 offset:16
; GCN-DAG: buffer_load_dword [[LOAD3:v[0-9]+]], off, s[0:3], s33 offset:20

; GCN-DAG: buffer_store_dword [[LOAD0]], off, s[0:3], s32 offset:4{{$}}
; GCN-DAG: buffer_store_dword [[LOAD1]], off, s[0:3], s32 offset:8
; GCN-DAG: buffer_store_dword [[LOAD2]], off, s[0:3], s32 offset:12
; GCN-DAG: buffer_store_dword [[LOAD3]], off, s[0:3], s32 offset:16

; GCN-DAG: buffer_load_dword [[LOAD4:v[0-9]+]], off, s[0:3], s33 offset:24
; GCN-DAG: buffer_load_dword [[LOAD5:v[0-9]+]], off, s[0:3], s33 offset:28
; GCN-DAG: buffer_load_dword [[LOAD6:v[0-9]+]], off, s[0:3], s33 offset:32
; GCN-DAG: buffer_load_dword [[LOAD7:v[0-9]+]], off, s[0:3], s33 offset:36

; GCN-DAG: buffer_store_dword [[LOAD4]], off, s[0:3], s32 offset:20
; GCN-DAG: buffer_store_dword [[LOAD5]], off, s[0:3], s32 offset:24
; GCN-DAG: buffer_store_dword [[LOAD6]], off, s[0:3], s32 offset:28
; GCN-DAG: buffer_store_dword [[LOAD7]], off, s[0:3], s32 offset:32


; GCN: s_swappc_b64
; GCN-NOT: s_sub_u32 s32
; GCN: s_endpgm
define amdgpu_kernel void @call_void_func_byval_struct_kernel() #1 {
entry:
  %arg0 = alloca %struct.ByValStruct, align 4, addrspace(5)
  %arg1 = alloca %struct.ByValStruct, align 4, addrspace(5)
  %tmp = bitcast %struct.ByValStruct addrspace(5)* %arg0 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp)
  %tmp1 = bitcast %struct.ByValStruct addrspace(5)* %arg1 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  store volatile i32 9, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  store volatile i32 13, i32 addrspace(5)* %arrayidx2, align 4
  call void @void_func_byval_struct(%struct.ByValStruct addrspace(5)* byval nonnull align 4 %arg0, %struct.ByValStruct addrspace(5)* byval nonnull align 4 %arg1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp)
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_align8:
; GCN: s_mov_b32 s5, s32
; GCN: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-NOT: s32
; GCN: buffer_store_dword [[LOAD0]], off, s[0:3], s5 offset:8{{$}}
; GCN-NOT: s32

; GCN: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s5 offset:24{{$}}
; GCN-NOT: s32
; GCN: buffer_store_dword [[LOAD1]], off, s[0:3], s5 offset:24{{$}}
; GCN-NOT: s32
define void @void_func_byval_struct_align8(%struct.ByValStruct addrspace(5)* byval noalias nocapture align 8 %arg0, %struct.ByValStruct addrspace(5)* byval noalias nocapture align 8 %arg1) #1 {
entry:
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  %tmp = load volatile i32, i32 addrspace(5)* %arrayidx, align 8
  %add = add nsw i32 %tmp, 1
  store volatile i32 %add, i32 addrspace(5)* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  %tmp1 = load volatile i32, i32 addrspace(5)* %arrayidx2, align 8
  %add3 = add nsw i32 %tmp1, 2
  store volatile i32 %add3, i32 addrspace(5)* %arrayidx2, align 8
  store volatile i32 9, i32 addrspace(1)* null, align 4
  ret void
}

; Make sure the byval alignment is respected in the call frame setup
; GCN-LABEL: {{^}}call_void_func_byval_struct_align8_kernel:
; GCN: s_mov_b32 s33, s7
; GCN: s_add_u32 s32, s33, 0xc00{{$}}

; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 [[THIRTEEN:v[0-9]+]], 13
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s33 offset:8
; GCN: buffer_store_dword [[THIRTEEN]], off, s[0:3], s33 offset:24

; GCN-NOT: s_add_u32 s32, s32, 0x800

; GCN-DAG: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s33 offset:8
; GCN-DAG: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s33 offset:12
; GCN-DAG: buffer_load_dword [[LOAD2:v[0-9]+]], off, s[0:3], s33 offset:16
; GCN-DAG: buffer_load_dword [[LOAD3:v[0-9]+]], off, s[0:3], s33 offset:20

; GCN-DAG: buffer_store_dword [[LOAD0]], off, s[0:3], s32 offset:8{{$}}
; GCN-DAG: buffer_store_dword [[LOAD1]], off, s[0:3], s32 offset:12
; GCN-DAG: buffer_store_dword [[LOAD2]], off, s[0:3], s32 offset:16
; GCN-DAG: buffer_store_dword [[LOAD3]], off, s[0:3], s32 offset:20

; GCN-DAG: buffer_load_dword [[LOAD4:v[0-9]+]], off, s[0:3], s33 offset:24
; GCN-DAG: buffer_load_dword [[LOAD5:v[0-9]+]], off, s[0:3], s33 offset:28
; GCN-DAG: buffer_load_dword [[LOAD6:v[0-9]+]], off, s[0:3], s33 offset:32
; GCN-DAG: buffer_load_dword [[LOAD7:v[0-9]+]], off, s[0:3], s33 offset:36

; GCN-DAG: buffer_store_dword [[LOAD4]], off, s[0:3], s32 offset:24
; GCN-DAG: buffer_store_dword [[LOAD5]], off, s[0:3], s32 offset:28
; GCN-DAG: buffer_store_dword [[LOAD6]], off, s[0:3], s32 offset:32
; GCN-DAG: buffer_store_dword [[LOAD7]], off, s[0:3], s32 offset:36


; GCN: s_swappc_b64
; GCN-NOT: s_sub_u32 s32
; GCN: s_endpgm
define amdgpu_kernel void @call_void_func_byval_struct_align8_kernel() #1 {
entry:
  %arg0 = alloca %struct.ByValStruct, align 8, addrspace(5)
  %arg1 = alloca %struct.ByValStruct, align 8, addrspace(5)
  %tmp = bitcast %struct.ByValStruct addrspace(5)* %arg0 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp)
  %tmp1 = bitcast %struct.ByValStruct addrspace(5)* %arg1 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  store volatile i32 9, i32 addrspace(5)* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  store volatile i32 13, i32 addrspace(5)* %arrayidx2, align 8
  call void @void_func_byval_struct_align8(%struct.ByValStruct addrspace(5)* byval nonnull align 8 %arg0, %struct.ByValStruct addrspace(5)* byval nonnull align 8 %arg1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp)
  ret void
}

; GCN-LABEL: {{^}}call_void_func_byval_struct_align8_func:
; GCN: s_mov_b32 s5, s32
; GCN-DAG: s_add_u32 s32, s32, 0xc00{{$}}
; GCN-DAG: v_writelane_b32

; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 [[THIRTEEN:v[0-9]+]], 13

; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s5 offset:8
; GCN-DAG: buffer_store_dword [[THIRTEEN]], off, s[0:3], s5 offset:24

; GCN: buffer_load_dword [[LOAD4:v[0-9]+]], off, s[0:3], s5 offset:24
; GCN: buffer_load_dword [[LOAD5:v[0-9]+]], off, s[0:3], s5 offset:28
; GCN: buffer_load_dword [[LOAD6:v[0-9]+]], off, s[0:3], s5 offset:32
; GCN: buffer_load_dword [[LOAD7:v[0-9]+]], off, s[0:3], s5 offset:36

; GCN-DAG: buffer_store_dword [[LOAD4]], off, s[0:3], s32 offset:24
; GCN-DAG: buffer_store_dword [[LOAD5]], off, s[0:3], s32 offset:28
; GCN-DAG: buffer_store_dword [[LOAD6]], off, s[0:3], s32 offset:32
; GCN-DAG: buffer_store_dword [[LOAD7]], off, s[0:3], s32 offset:36

; GCN-DAG: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s5 offset:8
; GCN-DAG: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s5 offset:12
; GCN-DAG: buffer_load_dword [[LOAD2:v[0-9]+]], off, s[0:3], s5 offset:16
; GCN-DAG: buffer_load_dword [[LOAD3:v[0-9]+]], off, s[0:3], s5 offset:20

; GCN-NOT: s_add_u32 s32, s32, 0x800

; GCN-DAG: buffer_store_dword [[LOAD0]], off, s[0:3], s32 offset:8{{$}}
; GCN-DAG: buffer_store_dword [[LOAD1]], off, s[0:3], s32 offset:12
; GCN-DAG: buffer_store_dword [[LOAD2]], off, s[0:3], s32 offset:16
; GCN-DAG: buffer_store_dword [[LOAD3]], off, s[0:3], s32 offset:20



; GCN: s_swappc_b64
; GCN-NOT: v_readlane_b32 s32
; GCN: v_readlane_b32
; GCN-NOT: v_readlane_b32 s32

; GCN-NOT: s_sub_u32 s32, s32, 0x800

; GCN: s_sub_u32 s32, s32, 0xc00{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @call_void_func_byval_struct_align8_func() #0 {
entry:
  %arg0 = alloca %struct.ByValStruct, align 8, addrspace(5)
  %arg1 = alloca %struct.ByValStruct, align 8, addrspace(5)
  %tmp = bitcast %struct.ByValStruct addrspace(5)* %arg0 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp)
  %tmp1 = bitcast %struct.ByValStruct addrspace(5)* %arg1 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  store volatile i32 9, i32 addrspace(5)* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  store volatile i32 13, i32 addrspace(5)* %arrayidx2, align 8
  call void @void_func_byval_struct_align8(%struct.ByValStruct addrspace(5)* byval nonnull align 8 %arg0, %struct.ByValStruct addrspace(5)* byval nonnull align 8 %arg1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp)
  ret void
}

; GCN-LABEL: {{^}}call_void_func_byval_struct_kernel_no_frame_pointer_elim:
define amdgpu_kernel void @call_void_func_byval_struct_kernel_no_frame_pointer_elim() #2 {
entry:
  %arg0 = alloca %struct.ByValStruct, align 4, addrspace(5)
  %arg1 = alloca %struct.ByValStruct, align 4, addrspace(5)
  %tmp = bitcast %struct.ByValStruct addrspace(5)* %arg0 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp)
  %tmp1 = bitcast %struct.ByValStruct addrspace(5)* %arg1 to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  store volatile i32 9, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  store volatile i32 13, i32 addrspace(5)* %arrayidx2, align 4
  call void @void_func_byval_struct(%struct.ByValStruct addrspace(5)* byval nonnull align 4 %arg0, %struct.ByValStruct addrspace(5)* byval nonnull align 4 %arg1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp1)
  call void @llvm.lifetime.end.p5i8(i64 32, i8 addrspace(5)* %tmp)
  ret void
}

declare void @external_void_func_void() #0

declare void @llvm.lifetime.start.p5i8(i64, i8 addrspace(5)* nocapture) #3
declare void @llvm.lifetime.end.p5i8(i64, i8 addrspace(5)* nocapture) #3

attributes #0 = { nounwind }
attributes #1 = { noinline norecurse nounwind }
attributes #2 = { nounwind norecurse "no-frame-pointer-elim"="true" }
