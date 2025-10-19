// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_store_1d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_1d(ptr %0, ptr addrspace(3) %1, i32 %2, i64 %3) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d(ptr addrspace(3) %1, ptr %0, i32 %2, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d(ptr addrspace(3) %1, ptr %0, i32 %2, i64 %3, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0] : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0] l2_cache_hint=%ch : !llvm.ptr, !llvm.ptr<3>

  llvm.return
}

llvm.func @tma_store_2d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_2d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i64 %4) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i64 %4, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1] : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1] l2_cache_hint=%ch : !llvm.ptr, !llvm.ptr<3>

  llvm.return
}

llvm.func @tma_store_3d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_3d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i64 %5) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i64 %5, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.3d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.3d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i64 %5, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2] : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2] l2_cache_hint=%ch : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2] {mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2] l2_cache_hint=%ch {mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_4d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_4d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.4d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.4d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3] : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3] l2_cache_hint=%ch : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3] {mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3] l2_cache_hint=%ch {mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_5d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_5d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.5d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.5d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] l2_cache_hint=%ch : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] {mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] l2_cache_hint=%ch {mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_scatter(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_scatter(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.scatter4.2d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.tile.scatter4.2d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] {mode = #nvvm.tma_store_mode<tile_scatter4>}: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] l2_cache_hint=%ch {mode = #nvvm.tma_store_mode<tile_scatter4>}: !llvm.ptr, !llvm.ptr<3>

  llvm.return
}
