; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) writeonly, ptr addrspace(3), ptr readonly, i32, i16, i64, i1 immarg, i1 immarg, i32 immarg range(i32 0, 3), i32 immarg range(i32 0, 6))

define void @test_cp_async_bulk_tensor_g2s_tile_1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch) {
  ; CHECK: immarg value 3 for arg 8 out of range [0,3)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch, i1 0, i1 0, i32 3, i32 0)

  ; CHECK: immarg value -1 for arg 8 out of range [0,3)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch, i1 0, i1 0, i32 -1, i32 0)

  ; CHECK: immarg value 6 for arg 9 out of range [0,6)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch, i1 0, i1 0, i32 0, i32 6)

  ; CHECK: immarg value -1 for arg 9 out of range [0,6)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch, i1 0, i1 0, i32 0, i32 -1)

  ret void
}
