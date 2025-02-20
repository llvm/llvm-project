; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Interleave loads and stores to fit into 9 VGPR limit.
; This requires to avoid load/store clustering.

; Reschedule the second scheduling region without clustering while
; the first region is skipped.

; GCN: global_load_dwordx4
; GCN: global_store_dwordx4
; GCN: global_load_dwordx4
; GCN: global_store_dwordx4
; GCN: global_load_dwordx4
; GCN: global_store_dwordx4
; GCN: NumVgprs: {{[0-9]$}}
; GCN: ScratchSize: 0{{$}}

define amdgpu_kernel void @load_store_max_9vgprs(ptr addrspace(1) nocapture noalias readonly %arg, ptr addrspace(1) nocapture noalias %arg1, i1 %cnd) #1 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %base = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg, i32 %id
  br i1 %cnd, label %bb1, label %bb2

bb1:
  %tmp = getelementptr inbounds <4 x i32>, ptr addrspace(1) %base, i32 1
  %tmp2 = load <4 x i32>, ptr addrspace(1) %tmp, align 4
  %tmp3 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %base, i32 3
  %tmp4 = load <4 x i32>, ptr addrspace(1) %tmp3, align 4
  %tmp5 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %base, i32 5
  %tmp6 = load <4 x i32>, ptr addrspace(1) %tmp5, align 4
  store <4 x i32> %tmp2, ptr addrspace(1) %arg1, align 4
  %tmp7 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg1, i64 3
  store <4 x i32> %tmp4, ptr addrspace(1) %tmp7, align 4
  %tmp8 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %arg1, i64 5
  store <4 x i32> %tmp6, ptr addrspace(1) %tmp8, align 4
  br label %bb2

bb2:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { "amdgpu-num-vgpr"="9" "amdgpu-flat-work-group-size"="1024,1024" }
