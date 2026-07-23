// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_bulk_prefetch(%src : !llvm.ptr<1>, %size : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_bulk_prefetch(ptr addrspace(1) %0, i32 %1, i64 %2) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.prefetch.L2(ptr addrspace(1) %0, i32 %1, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.prefetch.L2(ptr addrspace(1) %0, i32 %1, i64 %2, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.prefetch %src, %size : !llvm.ptr<1>
  nvvm.cp.async.bulk.prefetch %src, %size l2_cache_hint = %ch : !llvm.ptr<1>
  llvm.return
}

llvm.func @tma_prefetch_1d(%tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_prefetch_1d(ptr %0, i32 %1, i64 %2) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %0, i32 %1, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %0, i32 %1, i64 %2, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

llvm.func @tma_prefetch_2d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_prefetch_2d(ptr %0, i32 %1, i32 %2, i64 %3) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %0, i32 %1, i32 %2, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %0, i32 %1, i32 %2, i64 %3, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

llvm.func @tma_prefetch_3d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %off0 : i16, %off1 : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_prefetch_3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i16 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %0, i32 %1, i32 %2, i32 %3, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %0, i32 %1, i32 %2, i32 %3, i64 %6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i64 %6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i16 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i16 %5, i64 %6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.128.3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i16 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.128.3d(ptr %0, i32 %1, i32 %2, i32 %3, i16 %4, i16 %5, i64 %6, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] l2_cache_hint = %ch : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  llvm.return
}

llvm.func @tma_prefetch_4d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %off0 : i16, %off1 : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_prefetch_4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i64 %7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 %7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 %7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.128.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.128.4d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i16 %5, i16 %6, i64 %7, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  llvm.return
}

llvm.func @tma_prefetch_5d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %off0 : i16, %off1 : i16, %off2 : i16, %ch : i64) {
  // CHECK-LABEL: define void @tma_prefetch_5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %9, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i16 %8, i64 %9, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 %9, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.128.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.w.128.5d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i16 %6, i16 %7, i64 %9, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1, %off2] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1, %off2] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr

  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  llvm.return
}

llvm.func @tma_prefetch_gather4_2d(%tma_desc : !llvm.ptr, %x0 : i32, %y1 : i32, %y2 : i32, %y3 : i32, %y4 : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_prefetch_gather4_2d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.gather4.2d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.gather4.2d(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, i1 true)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%x0, %y1, %y2, %y3, %y4] {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%x0, %y1, %y2, %y3, %y4] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr
  llvm.return
}
