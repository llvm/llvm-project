// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_store_3d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_3d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i64 %5) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.3d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.3d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i64 %5, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2]  mode = im2col_w: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2] l2_cache_hint=%ch  mode = im2col_w: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_4d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_4d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.4d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.4d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3]  mode = im2col_w: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3] l2_cache_hint=%ch  mode = im2col_w: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_5d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %crd4: i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_5d(ptr %0, ptr addrspace(3) %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.5d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.s2g.im2col.w.5d(ptr addrspace(3) %1, ptr %0, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4]  mode = im2col_w: !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3,%crd4] l2_cache_hint=%ch  mode = im2col_w: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}
