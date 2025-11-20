; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_100a -o /dev/null 2>&1 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) writeonly, ptr addrspace(3), ptr readonly, i32, i16, i64, i1 immarg, i1 immarg, i32 immarg range(i32 0, 3))

define void @test_cp_async_bulk_tensor_g2s_tile_1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch) {
  ; CHECK: immarg value 3 out of range [0, 3)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch, i1 0, i1 0, i32 3)

  ; CHECK: immarg value -1 out of range [0, 3)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch, i1 0, i1 0, i32 -1)

  ret void
}
