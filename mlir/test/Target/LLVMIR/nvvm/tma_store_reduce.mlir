// RUN: mlir-translate -mlir-to-llvmir -split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: define void @tma_store_reduce_1d(
llvm.func @tma_store_reduce_1d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.1d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.1d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<tile>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_2d(
llvm.func @tma_store_reduce_2d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.2d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.2d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_3d_tile(
llvm.func @tma_store_reduce_3d_tile(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.3d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: define void @tma_store_reduce_3d_im2col(
llvm.func @tma_store_reduce_3d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.im2col.3d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<im2col>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.im2col.3d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<im2col>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_4d_tile(
llvm.func @tma_store_reduce_4d_tile(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.4d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: define void @tma_store_reduce_4d_im2col(
llvm.func @tma_store_reduce_4d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.im2col.4d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<im2col>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.im2col.4d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<im2col>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

// CHECK-LABEL: define void @tma_store_reduce_5d_tile(
llvm.func @tma_store_reduce_5d_tile(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.5d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i32 %[[D4:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.tile.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<add>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<min>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<max>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<inc>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<dec>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<and>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<or>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<xor>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: define void @tma_store_reduce_5d_im2col(
llvm.func @tma_store_reduce_5d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.im2col.5d(ptr addrspace(3) %[[SRC:.*]], ptr %[[DST:.*]], i32 %[[D0:.*]], i32 %[[D1:.*]], i32 %[[D2:.*]], i32 %[[D3:.*]], i32 %[[D4:.*]], i64 %[[CH:.*]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 %[[CH]], i1 true)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<im2col>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.add.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.min.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.max.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.inc.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.dec.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.and.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.or.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  // CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.xor.im2col.5d(ptr addrspace(3) %[[SRC]], ptr %[[DST]], i32 %[[D0]], i32 %[[D1]], i32 %[[D2]], i32 %[[D3]], i32 %[[D4]], i64 undef, i1 false)
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<min>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<max>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<inc>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<dec>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<or>, mode = #nvvm.tma_store_mode<im2col>}  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] {redKind = #nvvm.tma_redux_kind<xor>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}
