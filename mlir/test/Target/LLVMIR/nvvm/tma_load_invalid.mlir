// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @tma_load_1d_im2col(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_0d(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<7>, %bar: !llvm.ptr<3>) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[] : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_gather(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %ch : i64) {
  // expected-error @below {{Gather4 mode expects 5 coordinates}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0,%crd1,%crd2,%crd3] l2_cache_hint=%ch {mode = #nvvm.tma_load_mode<tile_gather4>}: !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_asm_im2col(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<7>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %wHalo: i16, %wOffset: i16, %p : i1) {
  // expected-error @below {{Inline-ptx lowering supported only for Tile/Im2col mode.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] predicate=%p {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}
