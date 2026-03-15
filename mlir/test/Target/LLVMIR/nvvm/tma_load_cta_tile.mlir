// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_load_1d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<3>, %bar: !llvm.ptr<3>, %crd0: i32, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_1d_all_tile(ptr %0, ptr addrspace(3) %1, ptr addrspace(3) %2, i32 %3, i64 %4) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.1d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.1d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i64 %4, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0] l2_cache_hint = %cacheHint {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_2d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<3>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_2d_all_tile(ptr %0, ptr addrspace(3) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i64 %5) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.2d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.2d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i64 %5, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1] l2_cache_hint = %cacheHint {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_3d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<3>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_3d_all_tile(ptr %0, ptr addrspace(3) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.3d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.3d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i64 %6, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2] l2_cache_hint = %cacheHint {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_4d_all_tile(%tma: !llvm.ptr, %dest: !llvm.ptr<3>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_4d_all_tile(ptr %0, ptr addrspace(3) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.4d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.4d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3] l2_cache_hint = %cacheHint {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_5d_all(%tma: !llvm.ptr, %dest: !llvm.ptr<3>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_5d_all(ptr %0, ptr addrspace(3) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.5d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.5d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%crd0, %crd1, %crd2, %crd3, %crd4] l2_cache_hint = %cacheHint {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

llvm.func @tma_load_2d_tile_gather4(%tma: !llvm.ptr, %dest: !llvm.ptr<3>, %bar: !llvm.ptr<3>, %row0: i32, %col0: i32, %col1: i32, %col2: i32, %col3: i32, %cacheHint: i64) {
  // CHECK-LABEL: define void @tma_load_2d_tile_gather4(ptr %0, ptr addrspace(3) %1, ptr addrspace(3) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.gather4.2d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.gather4.2d(ptr addrspace(3) %1, ptr addrspace(3) %2, ptr %0, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i64 %8, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr<3>, !llvm.ptr
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma, %bar, box[%row0, %col0, %col1, %col2, %col3] l2_cache_hint = %cacheHint {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}