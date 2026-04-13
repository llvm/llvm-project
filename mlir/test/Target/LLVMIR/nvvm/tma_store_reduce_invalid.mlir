// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @tma_reduce_0d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %ch : i64) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[] {redKind = #nvvm.tma_redux_kind<add>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

llvm.func @tma_reduce_2d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

llvm.func @tma_store_reduce_scatter(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // expected-error @below {{Scatter mode unsupported for CpAsyncBulkTensorReduceOp}}
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<tile_scatter4>} : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}
