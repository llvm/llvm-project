// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @nvvm_fence_proxy_acquire(%addr : !llvm.ptr, %size : i32) {
  // expected-error @below {{'nvvm.fence.proxy.acquire' op uni-directional proxies only support generic for from_proxy attribute}}
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cta> %addr, %size from_proxy=#nvvm.proxy_kind<tensormap> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_release() {
  // expected-error @below {{'nvvm.fence.proxy.release' op uni-directional proxies only support generic for from_proxy attribute}}
  nvvm.fence.proxy.release #nvvm.mem_scope<cta> from_proxy=#nvvm.proxy_kind<tensormap> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_acquire(%addr : !llvm.ptr, %size : i32) {
  // expected-error @below {{'nvvm.fence.proxy.acquire' op uni-directional proxies only support tensormap for to_proxy attribute}}
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cta> %addr, %size  from_proxy=#nvvm.proxy_kind<generic> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_release() {
  // expected-error @below {{'nvvm.fence.proxy.release' op uni-directional proxies only support tensormap for to_proxy attribute}}
  nvvm.fence.proxy.release #nvvm.mem_scope<cta> from_proxy=#nvvm.proxy_kind<generic> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @tma_prefetch_0d(%tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[] : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_2d_im2col(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %off0 : i16, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] im2col[%off0] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_5d_im2col(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %off0 : i16, %off1 : i16, %off2 : i16, %ch : i64) {
  // expected-error @below {{im2col offsets must be 2 less than number of coordinates}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] : !llvm.ptr
  llvm.return
}
