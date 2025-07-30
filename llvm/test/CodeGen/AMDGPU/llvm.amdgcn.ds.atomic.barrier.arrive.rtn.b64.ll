; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck --check-prefix=GCN %s

declare i64 @llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64(ptr addrspace(3), i64)

; GCN-LABEL: {{^}}test_ds_atomic_barrier_arrive_rtn_b64:
; GCN: ds_atomic_barrier_arrive_rtn_b64 v[{{[0-9:]+}}], v2, v[0:1]{{$}}
; GCN: s_wait_dscnt 0x0
; GCN: flat_store_b64
define void @test_ds_atomic_barrier_arrive_rtn_b64(i64 %data, ptr addrspace(3) %bar, ptr %out) {
entry:
  %ret = call i64 @llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64(ptr addrspace(3) %bar, i64 %data)
  store i64 %ret, ptr %out
  ret void
}

; GCN-LABEL: {{^}}test_ds_atomic_barrier_arrive_rtn_b64_off:
; GCN: ds_atomic_barrier_arrive_rtn_b64 v[{{[0-9:]+}}], v0, v[{{[0-9:]+}}] offset:8184{{$}}
; GCN: s_wait_dscnt 0x0
; GCN: flat_store_b64
define void @test_ds_atomic_barrier_arrive_rtn_b64_off(ptr addrspace(3) %in, ptr %out) {
entry:
  %bar = getelementptr i64, ptr addrspace(3) %in, i32 1023
  %ret = call i64 @llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64(ptr addrspace(3) %bar, i64 512)
  store i64 %ret, ptr %out
  ret void
}
