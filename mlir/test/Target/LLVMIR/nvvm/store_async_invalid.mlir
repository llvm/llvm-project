// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @st_async_global_mbarrier(%addr: !llvm.ptr<1>, %value: i32, %mbar: !llvm.ptr<7>) {
  // expected-error @below {{mbarrier is not supported for global address space}}
  nvvm.store.async %addr, %value, mbarrier = %mbar {scope = #nvvm.async_store_scope<sys>} : !llvm.ptr<1>, i32, !llvm.ptr<7>
  llvm.return
}

// -----

llvm.func @st_async_global_i128(%addr: !llvm.ptr<1>, %value: i128) {
  // expected-error @below {{only 8, 16, 32, and 64 bit values are supported for global address space}}
  nvvm.store.async %addr, %value {scope = #nvvm.async_store_scope<sys>} : !llvm.ptr<1>, i128
  llvm.return
}

// -----

llvm.func @st_async_global_no_scope(%addr: !llvm.ptr<1>, %value: i32) {
  // expected-error @below {{scope must be set for async store to global address space}}
  nvvm.store.async %addr, %value : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @st_async_global_mmio_non_sys(%addr: !llvm.ptr<1>, %value: i32) {
  // expected-error @below {{mmio is only supported for SYS scope}}
  nvvm.store.async %addr, %value {scope = #nvvm.async_store_scope<gpu>, is_mmio = true} : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @st_async_global_mmio_multimem(%addr: !llvm.ptr<1>, %value: i32) {
  // expected-error @below {{multimem is not supported for mmio}}
  nvvm.store.async %addr, %value {scope = #nvvm.async_store_scope<sys>, is_mmio = true, is_multimem = true} : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @st_async_shared_cluster_i8(%addr: !llvm.ptr<7>, %value: i8, %mbar: !llvm.ptr<7>) {
  // expected-error @below {{only 32, 64, and 128 bit values are supported for shared cluster address space}}
  nvvm.store.async %addr, %value, mbarrier = %mbar : !llvm.ptr<7>, i8, !llvm.ptr<7>
  llvm.return
}

// -----

llvm.func @st_async_shared_cluster_multimem(%addr: !llvm.ptr<7>, %value: i32, %mbar: !llvm.ptr<7>) {
  // expected-error @below {{multimem and mmio are not supported for shared cluster address space}}
  nvvm.store.async %addr, %value, mbarrier = %mbar {is_multimem = true} : !llvm.ptr<7>, i32, !llvm.ptr<7>
  llvm.return
}

// -----

llvm.func @st_async_shared_cluster_scope(%addr: !llvm.ptr<7>, %value: i32, %mbar: !llvm.ptr<7>) {
  // expected-error @below {{scope is not supported for async store to shared cluster address space}}
  nvvm.store.async %addr, %value, mbarrier = %mbar {scope = #nvvm.async_store_scope<sys>} : !llvm.ptr<7>, i32, !llvm.ptr<7>
  llvm.return
}
