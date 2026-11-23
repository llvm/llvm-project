// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

llvm.func @fence_sync_restrict() {
  // expected-error @below {{attribute 'order' failed to satisfy constraint: NVVM Memory Ordering kind whose value is one of {acquire, release}}}
  nvvm.fence.sync_restrict <weak>
  llvm.return
}

// -----

llvm.func @fence_sync_restrict() {
  // expected-error @below {{attribute 'order' failed to satisfy constraint: NVVM Memory Ordering kind whose value is one of {acquire, release}}}
  nvvm.fence.sync_restrict <mmio>
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: Proxy kind whose value is none of {tensormap, generic}}}
  nvvm.fence.proxy <tensormap>
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: Proxy kind whose value is none of {tensormap, generic}}}
  nvvm.fence.proxy <generic>
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{async_shared fence requires space attribute}}
  nvvm.fence.proxy <async.shared>
  llvm.return
}

// -----

llvm.func @fence_proxy() {
  // expected-error @below {{only async_shared fence can have space attribute}}
  nvvm.fence.proxy <alias>, space = <cta>
  llvm.return
}

// -----

llvm.func @fence_proxy_release() {
  // expected-error @below {{uni-directional proxies only support generic for from_proxy attribute}}
  nvvm.fence.proxy.release cta from_proxy = <alias> to_proxy = <tensormap>
  llvm.return
}

// -----

llvm.func @fence_proxy_release() {
  // expected-error @below {{uni-directional proxies only support tensormap for to_proxy attribute}}
  nvvm.fence.proxy.release cta from_proxy = <generic> to_proxy = <async>
  llvm.return
}

// -----

llvm.func @fence_proxy_sync_restrict() {
  // expected-error @below {{attribute 'order' failed to satisfy constraint: NVVM Memory Ordering kind whose value is one of {acquire, release}}}
  nvvm.fence.proxy.sync_restrict <mmio>
  llvm.return
}

// -----

llvm.func @fence_proxy_sync_restrict() {
  // expected-error @below {{only async is supported for to_proxy attribute}}
  nvvm.fence.proxy.sync_restrict <acquire>
      from_proxy = <generic> to_proxy = <alias>
  llvm.return
}

// -----

llvm.func @fence_proxy_sync_restrict() {
  // expected-error @below {{only generic is support for from_proxy attribute}}
  nvvm.fence.proxy.sync_restrict <acquire>
      from_proxy = <tensormap> to_proxy = <async>
  llvm.return
}
