// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @tma_store_1d_im2col(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0] {mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>

  llvm.return
}

// -----

llvm.func @tma_store_0d(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[] : !llvm.ptr, !llvm.ptr<3>

  llvm.return
}

// -----

llvm.func @tma_store_scatter(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %ch : i64) {
  // expected-error @below {{Scatter4 mode expects 5 coordinates}}
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0,%crd1,%crd2,%crd3] l2_cache_hint=%ch {mode = #nvvm.tma_store_mode<tile_scatter4>}: !llvm.ptr, !llvm.ptr<3>

  llvm.return
}

// -----

llvm.func @tma_store_asm_ch(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %ch : i64, %p : i1) {
  // expected-error @below {{Inline-ptx lowering unsupported with L2 cache-hint.}}
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0] l2_cache_hint=%ch, predicate=%p : !llvm.ptr, !llvm.ptr<3>

  llvm.return
}

// -----

llvm.func @tma_store_asm_im2col(%tma_desc: !llvm.ptr, %src : !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %ch : i64, %p : i1) {
  // expected-error @below {{Inline-ptx lowering supported only for Tile mode.}}
  nvvm.cp.async.bulk.tensor.global.shared.cta %tma_desc, %src, box[%crd0, %crd1, %crd2], predicate=%p {mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>

  llvm.return
}
