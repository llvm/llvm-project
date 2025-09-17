// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @tma_prefetch_0d(%tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[] : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_2d_im2col(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %off0 : i16, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] im2col[%off0] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_5d_im2col(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %off0 : i16, %off1 : i16, %off2 : i16, %ch : i64) {
  // expected-error @below {{im2col offsets expected 3 (provided 2)}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_3d_im2col_w(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %off0 : i16) {
  // expected-error @below {{im2col offsets expected 2 (provided 1)}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2] im2col[%off0] {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_4d_im2col_w_128(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %off0 : i16) {
  // expected-error @below {{im2col offsets expected 2 (provided 1)}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3] im2col[%off0] {mode = #nvvm.tma_load_mode<im2col_w_128>} : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_gather4_3d(%tma_desc : !llvm.ptr, %d0 : i32) {
  // expected-error @below {{Gather4 mode expects 5 coordinates}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0] {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_gather4_2d(%tma_desc : !llvm.ptr, %x0 : i32, %y1 : i32, %y2 : i32, %y3 : i32, %y4 : i32, %off0 : i16, %ch : i64) {
  // expected-error @below {{im2col offsets expected 0 (provided 1)}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%x0, %y1, %y2, %y3, %y4] im2col[%off0] l2_cache_hint = %ch {mode = #nvvm.tma_load_mode<tile_gather4>} : !llvm.ptr
  llvm.return
}

