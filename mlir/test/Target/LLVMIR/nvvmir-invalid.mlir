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