// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that global variable declarations are omitted from being transformed to internal linkage by the LLVM-IR lowering,
// preventing module verification issues.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.requires = #omp.clause_requires<unified_shared_memory>} {
  // CHECK-DAG: @_QMtest_0Evar_decl_usm = external addrspace(1) global float
  // CHECK-DAG: @_QMtest_0Evar_decl_usm_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Evar_decl_usm() {addr_space = 1 : i32, omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} : f32

  llvm.func @test_usm_declare_target_declaration() attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_decl_usm : !llvm.ptr<1>
    %1 = llvm.addrspacecast %0 : !llvm.ptr<1> to !llvm.ptr
    // CHECK-DAG: %[[DECL_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_decl_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: %{{.*}} = load float, ptr %[[DECL_VAR]], align 4
    %2 = llvm.load %1 : !llvm.ptr -> f32
    llvm.return
  }
}
