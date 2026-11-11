// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_shared_cta_global_tile_override_addr(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
// CHECK-LABEL: define void @tma_shared_cta_global_tile_override_addr(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: ret void
// CHECK-NEXT: }

  // without cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_shared_cta_global_tile_scatter4_override_addr(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
// CHECK-LABEL: define void @tma_shared_cta_global_tile_scatter4_override_addr(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.scatter4.override.addr.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.scatter4.override.addr.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: ret void
// CHECK-NEXT: }

  // without cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] mode = tile_scatter4 : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch mode = tile_scatter4 : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_shared_cta_global_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
// CHECK-LABEL: define void @tma_shared_cta_global_im2col(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8) {
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %8, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %8, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: ret void
// CHECK-NEXT: }

  // without cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_shared_cta_global_im2col_w(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
// CHECK-LABEL: define void @tma_shared_cta_global_im2col_w(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8) {
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %8, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %8, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: ret void
// CHECK-NEXT: }

  // without cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_shared_cta_global_tile_override_addr_dim_stride(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
// CHECK-LABEL: define void @tma_shared_cta_global_tile_override_addr_dim_stride(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i32 %13, i32 %14, i16 %17, i32 %3, i32 %4, i32 %5, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i32 %13, i32 %14, i32 %15, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* flag_cache_hint= */ i1 false)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i32 %13, i32 %14, i16 %17, i32 %3, i32 %4, i32 %5, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i32 %13, i32 %14, i32 %15, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.override.addr.dim.stride.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* flag_cache_hint= */ i1 true)
// CHECK-NEXT: ret void
// CHECK-NEXT: }

  // without cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] tensor_size[%ts0, %ts1, %ts2] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] tensor_size[%ts0, %ts1, %ts2, %ts3] lower_stride[%lstrd0, %lstrd1, %lstrd2] upper_stride[%ustrd] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] tensor_size[%ts0, %ts1, %ts2, %ts3, %ts4] lower_stride[%lstrd0, %lstrd1, %lstrd2, %lstrd3] upper_stride[%ustrd] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] tensor_size[%ts0, %ts1, %ts2] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] tensor_size[%ts0, %ts1, %ts2, %ts3] lower_stride[%lstrd0, %lstrd1, %lstrd2] upper_stride[%ustrd] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] tensor_size[%ts0, %ts1, %ts2, %ts3, %ts4] lower_stride[%lstrd0, %lstrd1, %lstrd2, %lstrd3] upper_stride[%ustrd] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

 llvm.return
}
