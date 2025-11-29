// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @mbarrier_arrive_ret_check(%barrier: !llvm.ptr<7>) {
  // expected-error @below {{mbarrier in shared_cluster space cannot return any value}}
  %0 = nvvm.mbarrier.arrive %barrier : !llvm.ptr<7> -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_arrive_invalid_scope(%barrier: !llvm.ptr<7>) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %0 = nvvm.mbarrier.arrive %barrier {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<7> -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_arrive_drop_ret_check(%barrier: !llvm.ptr<7>) {
  // expected-error @below {{mbarrier in shared_cluster space cannot return any value}}
  %0 = nvvm.mbarrier.arrive_drop %barrier : !llvm.ptr<7> -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_arrive_drop_invalid_scope(%barrier: !llvm.ptr<7>) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %0 = nvvm.mbarrier.arrive_drop %barrier {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<7> -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_expect_tx_scope(%barrier: !llvm.ptr<7>, %tx_count: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  nvvm.mbarrier.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<7>, i32
  llvm.return
}

// -----

llvm.func @mbarrier_complete_tx_scope(%barrier: !llvm.ptr<3>, %tx_count: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  nvvm.mbarrier.complete_tx %barrier, %tx_count {scope = #nvvm.mem_scope<sys>} : !llvm.ptr<3>, i32
  llvm.return
}
