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

// -----

llvm.func @mbarrier_arr_expect_tx(%barrier: !llvm.ptr<3>, %tx_count: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %1 = nvvm.mbarrier.arrive.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<3>, i32 -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_arr_expect_tx_cluster(%barrier: !llvm.ptr<7>, %tx_count: i32) {
  // expected-error @below {{mbarrier in shared_cluster space cannot return any value}}
  %1 = nvvm.mbarrier.arrive.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<7>, i32 -> i64
  llvm.return
}

// -----

llvm.func @init_mbarrier_arrive_expect_tx_asm_ret(%barrier : !llvm.ptr<3>, %txcount : i32, %pred : i1) {
  // expected-error @below {{return-value is not supported when using predicate}}
  %1 = nvvm.mbarrier.arrive.expect_tx %barrier, %txcount, predicate = %pred : !llvm.ptr<3>, i32, i1 -> i64 
  llvm.return
}

// -----

llvm.func @init_mbarrier_arrive_expect_tx_asm_relaxed(%barrier : !llvm.ptr<3>, %txcount : i32, %pred : i1) {
  // expected-error @below {{mbarrier with relaxed semantics is not supported when using predicate}}
  nvvm.mbarrier.arrive.expect_tx %barrier, %txcount, predicate = %pred {relaxed = true} : !llvm.ptr<3>, i32, i1
  llvm.return
}

// -----

llvm.func @init_mbarrier_arrive_expect_tx_asm_cta(%barrier : !llvm.ptr<3>, %txcount : i32, %pred : i1) {
  // expected-error @below {{mbarrier scope must be CTA when using predicate}}
  nvvm.mbarrier.arrive.expect_tx %barrier, %txcount, predicate = %pred {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32, i1
  llvm.return
}

// -----

llvm.func @init_mbarrier_arrive_expect_tx_asm_cluster(%barrier : !llvm.ptr<7>, %txcount : i32, %pred : i1) {
  // expected-error @below {{mbarrier in shared_cluster space is not supported when using predicate}}
  nvvm.mbarrier.arrive.expect_tx %barrier, %txcount, predicate = %pred : !llvm.ptr<7>, i32, i1
  llvm.return
}

// -----

llvm.func @mbarrier_arr_drop_expect_tx(%barrier: !llvm.ptr<3>, %tx_count: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %1 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<3>, i32 -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_arr_drop_expect_tx_cluster(%barrier: !llvm.ptr<7>, %tx_count: i32) {
  // expected-error @below {{mbarrier in shared_cluster space cannot return any value}}
  %1 = nvvm.mbarrier.arrive_drop.expect_tx %barrier, %tx_count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<7>, i32 -> i64
  llvm.return
}

// -----

llvm.func @mbarrier_test_wait(%barrier: !llvm.ptr<3>, %phase: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %1 = nvvm.mbarrier.test.wait %barrier, %phase {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<3>, i32 -> i1
  llvm.return
}

// -----

llvm.func @mbarrier_try_wait(%barrier: !llvm.ptr<3>, %phase: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %1 = nvvm.mbarrier.try_wait %barrier, %phase {scope = #nvvm.mem_scope<sys>} : !llvm.ptr<3>, i32 -> i1
  llvm.return
}

// -----

llvm.func @mbarrier_try_wait_with_timelimit(%barrier: !llvm.ptr<3>, %phase: i32, %ticks: i32) {
  // expected-error @below {{mbarrier scope must be either CTA or Cluster}}
  %1 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {scope = #nvvm.mem_scope<gpu>} : !llvm.ptr<3>, i32, i32 -> i1
  llvm.return
}

