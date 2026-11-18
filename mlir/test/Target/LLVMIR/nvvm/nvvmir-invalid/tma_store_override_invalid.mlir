// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @tma_shared_to_global_tile_override_addr_coord_tensor_size_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Expected coordinates size to be equal to tensor size}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_tile_override_addr_lower_stride_tensor_size_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Expected lower_stride size to be equal to one less than tensor size}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_tile_override_addr_stride_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Expected lower_stride and upper_stride to be either both present or both absent}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_tile_override_addr_im2col_dim_stride(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Only tile mode supports override address with dim and stride}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] tensor_size[%ts0, %ts1, %ts2] mode = im2col : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_tile_override_addr_tile_scatter4_dim_stride(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Only tile mode supports override address with dim and stride}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3, %d4] tensor_size[%ts0, %ts1, %ts2, %ts3, %ts4] mode = tile_scatter4 : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_tile_override_addr_tile_scatter4_coord_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Mode tile scatter4 expects 5 coordinates}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2, %d3] mode = tile_scatter4 : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_tile_override_addr_lower_stride_no_tensor_size(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %lstrd0 : i32, %lstrd1 : i32, %ustrd : i16) {
  // expected-error @below {{Expected tensor_size to be present when lower_stride is provided}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd] : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_shared_to_global_im2col_w_override_addr_2d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1] mode = im2col_w : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

