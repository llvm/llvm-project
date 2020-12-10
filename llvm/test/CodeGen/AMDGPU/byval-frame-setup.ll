; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s

%struct.ByValStruct = type { [4 x i32] }
; Make sure the offset is folded and function's frame register is used
; rather than the global scratch wave offset.
; GCN-LABEL: {{^}}void_func_byval_struct_use_outside_entry_block:
; GCN-NOT: v_lshrrev_b32
; GCN-NOT: s_sub_u32

; GCN: s_and_saveexec_b64
; GCN: s_cbranch_execz [[BB1:BB[0-9]+_[0-9]+]]

; GCN: buffer_load_dword [[LOAD0:v[0-9]+]], off, s[0:3], s32{{$}}
; GCN-NOT: s32
; GCN: buffer_store_dword [[LOAD0]], off, s[0:3], s32{{$}}
; GCN-NOT: s32

; GCN: buffer_load_dword [[LOAD1:v[0-9]+]], off, s[0:3], s32 offset:16{{$}}
; GCN-NOT: s32
; GCN: buffer_store_dword [[LOAD1]], off, s[0:3], s32 offset:16{{$}}
; GCN-NOT: s32

; GCN: [[BB1]]
; GCN: s_or_b64 exec, exec
define hidden void @void_func_byval_struct_use_outside_entry_block(%struct.ByValStruct addrspace(5)* byval(%struct.ByValStruct) noalias nocapture align 4 %arg0, %struct.ByValStruct addrspace(5)* byval(%struct.ByValStruct) noalias nocapture align 4 %arg1, i1 %cond) #1 {
entry:
  br i1 %cond, label %bb0, label %bb1

bb0:
  %arrayidx = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg0, i32 0, i32 0, i32 0
  %tmp = load volatile i32, i32 addrspace(5)* %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  store volatile i32 %add, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.ByValStruct, %struct.ByValStruct addrspace(5)* %arg1, i32 0, i32 0, i32 0
  %tmp1 = load volatile i32, i32 addrspace(5)* %arrayidx2, align 4
  %add3 = add nsw i32 %tmp1, 2
  store volatile i32 %add3, i32 addrspace(5)* %arrayidx2, align 4
  store volatile i32 9, i32 addrspace(1)* null, align 4
  br label %bb1

bb1:
  ret void
}
declare hidden void @external_void_func_void() #0

declare void @llvm.lifetime.start.p5i8(i64, i8 addrspace(5)* nocapture) #3
declare void @llvm.lifetime.end.p5i8(i64, i8 addrspace(5)* nocapture) #3

attributes #0 = { nounwind }
attributes #1 = { noinline norecurse nounwind }
attributes #2 = { nounwind norecurse "frame-pointer"="all" }
