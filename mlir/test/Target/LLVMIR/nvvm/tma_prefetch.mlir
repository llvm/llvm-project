// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @tma_prefetch_1d
llvm.func @tma_prefetch_1d(%tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %0, i32 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %0, i32 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @tma_prefetch_2d
llvm.func @tma_prefetch_2d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @tma_prefetch_3d
llvm.func @tma_prefetch_3d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %off0 : i16, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] l2_cache_hint = %ch : !llvm.ptr

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i16 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i16 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @tma_prefetch_4d
llvm.func @tma_prefetch_4d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %off0 : i16, %off1 : i16, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch : !llvm.ptr

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0, %off1] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @tma_prefetch_5d
llvm.func @tma_prefetch_5d(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %off0 : i16, %off1 : i16, %off2 : i16, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch : !llvm.ptr

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, i64 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %0, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, i64 %{{.*}}, i1 true)
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1, %off2] : !llvm.ptr
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1, %off2] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}
