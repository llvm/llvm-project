// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @fence_sync_restrict() {
  // expected-error @below {{attribute 'order' failed to satisfy constraint: NVVM Memory Ordering kind whose value is one of {acquire, release}}}
  nvvm.fence.sync_restrict {order = #nvvm.mem_order<weak>}
  llvm.return
}

// -----

llvm.func @fence_sync_restrict() {
  // expected-error @below {{attribute 'order' failed to satisfy constraint: NVVM Memory Ordering kind whose value is one of {acquire, release}}}
  nvvm.fence.sync_restrict {order = #nvvm.mem_order<mmio>}
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{tensormap proxy is not a supported proxy kind}}
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<tensormap>}
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{generic proxy not a supported proxy kind}}
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<generic>}
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{async_shared fence requires space attribute}}
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>}
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{only async_shared fence can have space attribute}}
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<alias>, space = #nvvm.shared_space<cta>}
  llvm.return
}

// -----

llvm.func @fence_proxy_release() {
  // expected-error @below {{uni-directional proxies only support generic for from_proxy attribute}}
  nvvm.fence.proxy.release #nvvm.mem_scope<cta> from_proxy = #nvvm.proxy_kind<alias> to_proxy = #nvvm.proxy_kind<tensormap>
  llvm.return
}

// -----

llvm.func @fence_proxy_release() {
  // expected-error @below {{uni-directional proxies only support tensormap for to_proxy attribute}}
  nvvm.fence.proxy.release #nvvm.mem_scope<cta> from_proxy = #nvvm.proxy_kind<generic> to_proxy = #nvvm.proxy_kind<async>
  llvm.return
}

// -----

llvm.func @fence_proxy_sync_restrict() {
  // expected-error @below {{attribute 'order' failed to satisfy constraint: NVVM Memory Ordering kind whose value is one of {acquire, release}}}
  nvvm.fence.proxy.sync_restrict {order = #nvvm.mem_order<mmio>}
  llvm.return
}

// -----

llvm.func @fence_proxy_sync_restrict() {
  // expected-error @below {{only async is supported for to_proxy attribute}}
  nvvm.fence.proxy.sync_restrict {order = #nvvm.mem_order<acquire>, toProxy = #nvvm.proxy_kind<alias>,
                                  fromProxy = #nvvm.proxy_kind<generic>}
  llvm.return
}

// -----

llvm.func @fence_proxy_sync_restrict() {
  // expected-error @below {{only generic is support for from_proxy attribute}}
  nvvm.fence.proxy.sync_restrict {order = #nvvm.mem_order<acquire>, toProxy = #nvvm.proxy_kind<async>,
                                  fromProxy = #nvvm.proxy_kind<tensormap>}
  llvm.return
}
