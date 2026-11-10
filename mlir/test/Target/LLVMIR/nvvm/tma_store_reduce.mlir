// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @tma_store_reduce_1d(
llvm.func @tma_store_reduce_1d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch , reduction = xor : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = add mode = tile : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = min mode = tile : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = max mode = tile : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = inc mode = tile : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = dec mode = tile : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = and mode = tile : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = or mode = tile  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] , reduction = xor mode = tile : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_2d(
llvm.func @tma_store_reduce_2d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch , reduction = xor : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] , reduction = xor : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_3d_tile(
llvm.func @tma_store_reduce_3d_tile(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = xor : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = xor : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: define void @tma_store_reduce_3d_im2col(
llvm.func @tma_store_reduce_3d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = or mode = im2col  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = or mode = im2col  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_4d_tile(
llvm.func @tma_store_reduce_4d_tile(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = xor : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = xor : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: define void @tma_store_reduce_4d_im2col(
llvm.func @tma_store_reduce_4d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = or mode = im2col  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = or mode = im2col  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_5d_tile(
llvm.func @tma_store_reduce_5d_tile(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i32 %[[D4:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = xor : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = add : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = min : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = max : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = inc : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = dec : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = and : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = or  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = xor : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: define void @tma_store_reduce_5d_im2col(
llvm.func @tma_store_reduce_5d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i32 %[[D4:.*]], i64 %[[CH:.*]], /* red_op=add */ i32 0, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=min */ i32 1, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=max */ i32 2, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=inc */ i32 3, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=dec */ i32 4, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=and */ i32 5, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=or */ i32 6, i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], /* red_op=xor */ i32 7, i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = or mode = im2col  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 0, /* red_op=xor */ i32 7, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = add mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = min mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = max mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = inc mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = dec mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = and mode = im2col : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = or mode = im2col  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = xor mode = im2col : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}
