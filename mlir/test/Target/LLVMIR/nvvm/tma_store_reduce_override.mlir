// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_store_reduce_tile_override_addr(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_tile_override_addr(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 0, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i64 %18, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // without cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test min reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = min : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = min : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test max reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = max : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = max : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test inc reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = inc : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = inc : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test dec reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = dec : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = dec : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test and reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = and : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = and : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test or reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = or : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = or : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test xor reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0], reduction = xor : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] l2_cache_hint = %ch, reduction = xor : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_store_reduce_im2col_override_addr(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_im2col_override_addr(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // without cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3], reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4], reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch, reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch, reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test min reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test max reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test inc reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test dec reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test and reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test or reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = or mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = or mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test xor reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_store_reduce_im2col_w_override_addr(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_im2col_w_override_addr(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.override.addr.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // without cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3], reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4], reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch, reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch, reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test min reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test max reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test inc reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test dec reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test and reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test or reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = or mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = or mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test xor reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2], reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch, reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  llvm.return
}

llvm.func @tma_store_reduce_tile_override_addr_dim_stride(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_tile_override_addr_dim_stride(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i64 %18) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i32 %13, i32 %14, i16 %17, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i32 %13, i32 %14, i32 %15, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.3d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i32 %13, i32 %14, i16 %17, i32 %3, i32 %4, i32 %5, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.4d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i32 %13, i32 %14, i32 %15, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.5d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i16 %10, i16 %11, i16 %12, i32 %13, i32 %14, i32 %15, i32 %16, i16 %17, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %18, /* red_op=add */ i32 0, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=min */ i32 1, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=max */ i32 2, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=inc */ i32 3, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=dec */ i32 4, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=and */ i32 5, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=or */ i32 6, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 0, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.1d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i32 %3, i64 %18, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 0, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.override.addr.dim.stride.2d(ptr addrspace(3) %0, ptr %1, ptr addrspace(1) %2, i16 %8, i16 %9, i32 %13, i16 %17, i32 %3, i32 %4, i64 %18, /* red_op=xor */ i32 7, /* flag_cache_hint= */ i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // without cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] tensor_size[%ts0, %ts1, %ts2] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] tensor_size[%ts0, %ts1, %ts2, %ts3] lower_stride[%lstrd0, %lstrd1, %lstrd2] upper_stride[%ustrd], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] tensor_size[%ts0, %ts1, %ts2, %ts3, %ts4] lower_stride[%lstrd0, %lstrd1, %lstrd2, %lstrd3] upper_stride[%ustrd], reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // with cache hint
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] tensor_size[%ts0, %ts1, %ts2] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] tensor_size[%ts0, %ts1, %ts2, %ts3] lower_stride[%lstrd0, %lstrd1, %lstrd2] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] tensor_size[%ts0, %ts1, %ts2, %ts3, %ts4] lower_stride[%lstrd0, %lstrd1, %lstrd2, %lstrd3] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = add : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test min reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = min : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = min : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = min : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = min : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test max reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = max : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = max : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = max : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = max : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test inc reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = inc : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = inc : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = inc : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = inc : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test dec reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = dec : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = dec : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = dec : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = dec : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test and reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = and : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = and : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = and : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = and : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test or reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = or : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = or : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = or : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = or : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  // Test xor reduction
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0], reduction = xor : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] tensor_size[%ts0] l2_cache_hint = %ch, reduction = xor : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd], reduction = xor : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] l2_cache_hint = %ch, reduction = xor : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>

 llvm.return
}
