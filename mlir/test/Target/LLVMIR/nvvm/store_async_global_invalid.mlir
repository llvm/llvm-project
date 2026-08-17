// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @st_async_global_invalid_scope(%addr: !llvm.ptr<1>, %value: i32) {
  // expected-error @below {{scope must be either SYS or GPU}}
  nvvm.store.async.global %addr, %value {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @st_async_global_mmio_non_sys(%addr: !llvm.ptr<1>, %value: i32) {
  // expected-error @below {{mmio is only supported for SYS scope}}
  nvvm.store.async.global %addr, %value {scope = #nvvm.mem_scope<gpu>, mmio = true} : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @st_async_global_mmio_multimem(%addr: !llvm.ptr<1>, %value: i32) {
  // expected-error @below {{multimem is not supported with mmio}}
  nvvm.store.async.global %addr, %value {scope = #nvvm.mem_scope<sys>, mmio = true, multimem = true} : !llvm.ptr<1>, i32
  llvm.return
}
