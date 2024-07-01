; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80| FileCheck --check-prefixes=CHECK_PTX64 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 --nvptx-short-ptr| FileCheck --check-prefixes=CHECK_PTX_SHARED32 %s
; RUN: %if ptxas-12.3 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80| %ptxas-verify -arch=sm_90 %}
; RUN: %if ptxas-12.3 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 --nvptx-short-ptr| %ptxas-verify -arch=sm_90 %}

declare void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.1d(i32 %flags, ptr addrspace(3) %s, ptr %tm, i32 %d0, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.2d(i32 %flags, ptr addrspace(3) %s, ptr %tm, i32 %d0, i32 %d1, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.3d(i32 %flags, ptr addrspace(3) %s, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.4d(i32 %flags, ptr addrspace(3) %s, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch);
declare void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.5d(i32 %flags, ptr addrspace(3) %s, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch);

; CHECK-LABEL: cp_async_bulk_tensor_s2g_1d
define void @cp_async_bulk_tensor_s2g_1d(i32 %flag, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.1d(i32 0, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.1d(i32 1, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_s2g_2d
define void @cp_async_bulk_tensor_s2g_2d(i32 %flag, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.2d(i32 0, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.2d(i32 1, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_s2g_3d
define void @cp_async_bulk_tensor_s2g_3d(i32 %flag, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.3d(i32 0, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.3d(i32 1, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.global.shared::cta.im2col_no_offs.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.global.shared::cta.im2col_no_offs.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.3d(i32 4, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.3d.global.shared::cta.im2col_no_offs.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.3d.global.shared::cta.im2col_no_offs.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.3d(i32 5, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_s2g_4d
define void @cp_async_bulk_tensor_s2g_4d(i32 %flag, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.4d(i32 0, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.4d(i32 1, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.4d(i32 4, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.4d(i32 5, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_s2g_5d
define void @cp_async_bulk_tensor_s2g_5d(i32 %flag, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch) {
  ; CHECK_PTX64: cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.5d(i32 0, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.5d(i32 1, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.global.shared::cta.im2col_no_offs.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}];
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.global.shared::cta.im2col_no_offs.bulk_group [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.5d(i32 4, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 undef)

  ; CHECK_PTX64: cp.async.bulk.tensor.5d.global.shared::cta.im2col_no_offs.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  ; CHECK_PTX_SHARED32: cp.async.bulk.tensor.5d.global.shared::cta.im2col_no_offs.bulk_group.L2::cache_hint [%rd{{[0-9]+}}, {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}], [%r{{[0-9]+}}], %rd{{[0-9]+}};
  tail call void @llvm.nvvm.cp.async.bulk.tensor.smem.to.gmem.5d(i32 5, ptr addrspace(3) %src, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch)
  ret void
}
