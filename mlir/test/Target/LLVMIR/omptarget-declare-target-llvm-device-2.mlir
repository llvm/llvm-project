// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks that we correctly generate ref pointers with the correct type in USM mode
// for link and to clauses. And verifies we continue to make the correct replacement accesses
// within the target region.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.requires = #omp<clause_requires unified_shared_memory>} {
  // CHECK-DAG: @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Evar_to_usm() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Evar_enter_usm() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
    %0 = llvm.mlir.constant(2 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Evar_link_usm_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Evar_link_usm() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(3 : i32) : i32
    llvm.return %0 : i32
  }

  llvm.func @test_usm_declare_target() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to_usm : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter_usm : !llvm.ptr
    %2 = llvm.mlir.addressof @_QMtest_0Evar_link_usm : !llvm.ptr
    // CHECK-DAG: %[[TO_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 10, ptr %[[TO_VAR]], align 4
    // CHECK-DAG: %[[ENTER_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 20, ptr %[[ENTER_VAR]], align 4
    // CHECK-DAG: %[[LINK_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_link_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 30, ptr %[[LINK_VAR]], align 4
    %map0 = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %map1 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%map0 -> %arg0, %map1 -> %arg1, %map2 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %c10 = llvm.mlir.constant(10 : i32) : i32
      %c20 = llvm.mlir.constant(20 : i32) : i32
      %c30 = llvm.mlir.constant(30 : i32) : i32
      llvm.store %c10, %arg0 : i32, !llvm.ptr
      llvm.store %c20, %arg1 : i32, !llvm.ptr
      llvm.store %c30, %arg2 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}
