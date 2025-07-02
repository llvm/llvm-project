; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck --check-prefix=GCN %s

declare void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3))

; GCN-LABEL: {{^}}test_ds_atomic_async_barrier_arrive_b64:
; GCN: ds_atomic_async_barrier_arrive_b64 v0{{$}}
define void @test_ds_atomic_async_barrier_arrive_b64(ptr addrspace(3) %bar) {
entry:
  call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %bar)
  ret void
}

; GCN-LABEL: {{^}}test_ds_atomic_async_barrier_arrive_b64_off:
; GCN: ds_atomic_async_barrier_arrive_b64 v0 offset:8184{{$}}
define void @test_ds_atomic_async_barrier_arrive_b64_off(ptr addrspace(3) %in) {
entry:
  %bar = getelementptr i64, ptr addrspace(3) %in, i32 1023
  call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %bar)
  ret void
}
