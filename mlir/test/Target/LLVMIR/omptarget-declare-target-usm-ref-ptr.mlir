// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This tests the replacement of uses of a declare target global variable with
// the unified shared memory (USM) generated reference pointer in an explicit device function.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true, omp.requires = #omp<clause_requires unified_shared_memory>} {
  // CHECK-DAG: @_QMmEnx_vals_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMmEnx_vals() {addr_space = 1 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.zero : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define void @_QMmPget_dims_noarg(ptr %0)
  llvm.func @_QMmPget_dims_noarg(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    // CHECK: %[[REF_LOAD:.*]] = load ptr, ptr @_QMmEnx_vals_decl_tgt_ref_ptr, align 8
    // CHECK: %[[AS_CAST:.*]] = addrspacecast ptr %[[REF_LOAD]] to ptr addrspace(1)
    // CHECK: %[[VAL:.*]] = load i32, ptr addrspace(1) %[[AS_CAST]], align 4
    // CHECK: store i32 %[[VAL]], ptr %0, align 4
    %0 = llvm.mlir.addressof @_QMmEnx_vals : !llvm.ptr<1>
    %1 = llvm.load %0 : !llvm.ptr<1> -> i32
    llvm.store %1, %arg0 : i32, !llvm.ptr
    llvm.return
  }
}