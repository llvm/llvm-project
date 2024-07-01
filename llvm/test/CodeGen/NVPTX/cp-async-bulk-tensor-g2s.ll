; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80| FileCheck --check-prefixes=CHECK_PTX64 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 --nvptx-short-ptr| FileCheck --check-prefixes=CHECK_PTX_SHARED32 %s
; RUN: %if ptxas-12.3 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80| %ptxas-verify -arch=sm_90 %}
; RUN: %if ptxas-12.3 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 --nvptx-short-ptr| %ptxas-verify -arch=sm_90 %}

declare void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.1d(i32 %flags, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i16 %mc, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.2d(i32 %flags, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i16 %mc, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 %flags, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 %flags, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 %flags, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch);

; CHECK-LABEL: cp_async_bulk_tensor_g2s_1d
define void @cp_async_bulk_tensor_g2s_1d(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.1d(i32 0, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.1d(i32 1, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 undef, i64 %ch)
  ; CHECK_PTX64: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.1d(i32 2, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.1d(i32 3, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_2d
define void @cp_async_bulk_tensor_g2s_2d(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.2d(i32 0, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.2d(i32 1, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.2d(i32 2, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.2d(i32 3, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_3d_tile
define void @cp_async_bulk_tensor_g2s_3d_tile(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 0, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 1, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 2, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 3, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_3d_im2col
define void @cp_async_bulk_tensor_g2s_3d_im2col(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 4, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 5, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}}, %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}}, %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 6, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}}, %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}}, %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.3d(i32 7, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_4d_tile
define void @cp_async_bulk_tensor_g2s_4d_tile(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 0, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 1, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 2, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 3, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_4d_im2col
define void @cp_async_bulk_tensor_g2s_4d_im2col(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 4, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 5, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 6, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.4d(i32 7, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_5d_tile
define void @cp_async_bulk_tensor_g2s_5d_tile(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 0, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 1, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 2, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 3, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_g2s_5d_im2col
define void @cp_async_bulk_tensor_g2s_5d_im2col(i32 %flag, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 4, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 undef, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 5, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 undef, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 6, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%rd{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}}, %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%r{{[0-9]+}}], [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], {%rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}}, %rs{{[0-9]+}}, %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.gmem.to.smem.5d(i32 7, ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch)
  ret void
}
