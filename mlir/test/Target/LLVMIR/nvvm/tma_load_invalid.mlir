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
  // expected-error @below {{Predicate is supported only for Tile and Im2col modes.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] predicate=%p {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}
// -----

llvm.func @tma_load_cta_asm_im2col(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<3>, %bar: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %wHalo: i16, %wOffset: i16, %p : i1) {
  // expected-error @below {{Predicate is supported only for shared::cluster mode.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0, %crd1, %crd2] im2col[%wHalo, %wOffset] predicate=%p {isCTAOnly = true, mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_cta_0d(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<3>, %bar : !llvm.ptr<3>) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_cta_mc(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<3>, %bar : !llvm.ptr<3>, %crd0: i32, %ctamask : i16) {
  // expected-error @below {{Multicast is not supported with shared::cta mode.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0] multicast_mask = %ctamask {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}
// -----

llvm.func @tma_load_cta_cg(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<3>, %bar : !llvm.ptr<3>, %crd0: i32, %crd1: i32) {
  // expected-error @below {{CTAGroup is not supported with shared::cta mode.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0, %crd1] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_cta_with_7(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<7>, %bar : !llvm.ptr<3>, %crd0: i32, %crd1: i32) {
  // expected-error @below {{Shared::cta destination requires address-space 3.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0, %crd1] {isCTAOnly = true, mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_cluster_with_3(%tma_desc: !llvm.ptr, %dest : !llvm.ptr<3>, %bar : !llvm.ptr<3>, %crd0: i32, %crd1: i32) {
  // expected-error @below {{Shared::cluster destination requires address-space 7.}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tma_desc, %bar, box[%crd0, %crd1] {isCTAOnly = false, mode = #nvvm.tma_load_mode<tile>, group = #nvvm.cta_group<cta_1>} : !llvm.ptr<3>, !llvm.ptr

  llvm.return
}

// -----

llvm.func @tma_load_im2col_off(%tmaDescriptor: !llvm.ptr, %dest : !llvm.ptr<7>, %barrier: !llvm.ptr<3>, %crd0: i32, %crd1: i32, %crd2: i32, %crd3: i32, %off0: i16, %off1: i16, %ctamask : i16, %cacheHint : i64) {
  // expected-error @below {{im2col offsets expected 2 (provided 1)}}
  nvvm.cp.async.bulk.tensor.shared.cluster.global %dest, %tmaDescriptor,  %barrier, box[%crd0,%crd1,%crd2,%crd3] im2col[%off0] multicast_mask = %ctamask l2_cache_hint = %cacheHint {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr<7>, !llvm.ptr

  llvm.return
}
