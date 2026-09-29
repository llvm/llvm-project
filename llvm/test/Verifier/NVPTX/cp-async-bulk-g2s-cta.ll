; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.nvvm.cp.async.bulk.global.to.shared.cta(ptr addrspace(3) writeonly, ptr addrspace(3), ptr addrspace(1) readonly, i32, i32, i32, i64, i1 immarg, i1 immarg, i32 immarg range(i32 0, 6))
declare void @llvm.nvvm.cp.async.bulk.global.to.shared.cta.relaxed(ptr addrspace(3) writeonly, ptr addrspace(3), ptr addrspace(1) readonly, i32, i32, i32, i64, i1 immarg, i1 immarg, i32 immarg range(i32 0, 4), i32 immarg range(i32 0, 6))

define void @test_cp_async_bulk_g2s_cta(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr addrspace(1) %src, i32 %size, i64 %ch) {
  ; CHECK: flag_valid_pattern must be 0 (disabled) when ignore_oob is enabled
  call void @llvm.nvvm.cp.async.bulk.global.to.shared.cta(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr addrspace(1) %src, i32 %size, i32 0, i32 0, i64 %ch, i1 0, i1 1, i32 2)
  ret void
}

define void @test_cp_async_bulk_g2s_cta_relaxed(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr addrspace(1) %src, i32 %size, i64 %ch) {
  ; CHECK: flag_valid_pattern must be 0 (disabled) when ignore_oob is enabled
  call void @llvm.nvvm.cp.async.bulk.global.to.shared.cta.relaxed(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr addrspace(1) %src, i32 %size, i32 0, i32 0, i64 %ch, i1 1, i1 1, i32 0, i32 4)
  ret void
}
