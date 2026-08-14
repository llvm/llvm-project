// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests lowering to LLVM IR of all combinations of OpenMP declare target
// device_type (any/host/nohost) and capture clause (link/enter) when compiling
// for device, for both external and internal linkage global variables.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  // --- link any ---

  // CHECK-DAG: @ial = internal global float 0.000000e+00
  // CHECK-DAG: @ial_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global internal @ial() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @eal = internal global float 0.000000e+00
  // CHECK-DAG: @eal_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @eal() {addr_space = 0 : i32, dso_local, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // --- link host ---

  // CHECK-DAG: @ihl = internal global float 0.000000e+00
  // CHECK-DAG: @ihl_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global internal @ihl() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @ehl = internal global float 0.000000e+00
  // CHECK-DAG: @ehl_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @ehl() {addr_space = 0 : i32, dso_local, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // --- link nohost ---

  // CHECK-DAG: @inl = internal global float 0.000000e+00
  // CHECK-DAG: @inl_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global internal @inl() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @enl = internal global float 0.000000e+00
  // CHECK-DAG: @enl_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @enl() {addr_space = 0 : i32, dso_local, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // --- enter any ---

  // CHECK-DAG: @iae = global float 0.000000e+00
  llvm.mlir.global internal @iae() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @eae = dso_local global float 0.000000e+00
  llvm.mlir.global external @eae() {addr_space = 0 : i32, dso_local, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // --- enter host ---

  // CHECK-DAG: @ihe = external dso_local global float
  llvm.mlir.global internal @ihe() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @ehe = external dso_local global float
  llvm.mlir.global external @ehe() {addr_space = 0 : i32, dso_local, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // --- enter nohost ---

  // CHECK-DAG: @ine = global float 0.000000e+00
  llvm.mlir.global internal @ine() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  // CHECK-DAG: @ene = dso_local global float 0.000000e+00
  llvm.mlir.global external @ene() {addr_space = 0 : i32, dso_local, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} : f32 {
    %0 = llvm.mlir.zero : f32
    llvm.return %0 : f32
  }

  llvm.func @_QQmain() {
    %0 = llvm.mlir.addressof @ial : !llvm.ptr
    %1 = llvm.addrspacecast %0 : !llvm.ptr to !llvm.ptr
    %2 = llvm.mlir.addressof @ihl : !llvm.ptr
    %3 = llvm.addrspacecast %2 : !llvm.ptr to !llvm.ptr
    %4 = llvm.mlir.addressof @inl : !llvm.ptr
    %5 = llvm.addrspacecast %4 : !llvm.ptr to !llvm.ptr
    %6 = llvm.mlir.addressof @iae : !llvm.ptr
    %7 = llvm.addrspacecast %6 : !llvm.ptr to !llvm.ptr
    %8 = llvm.mlir.addressof @ihe : !llvm.ptr
    %9 = llvm.addrspacecast %8 : !llvm.ptr to !llvm.ptr
    %10 = llvm.mlir.addressof @ine : !llvm.ptr
    %11 = llvm.addrspacecast %10 : !llvm.ptr to !llvm.ptr
    %12 = llvm.mlir.addressof @eal : !llvm.ptr
    %13 = llvm.addrspacecast %12 : !llvm.ptr to !llvm.ptr
    %14 = llvm.mlir.addressof @ehl : !llvm.ptr
    %15 = llvm.addrspacecast %14 : !llvm.ptr to !llvm.ptr
    %16 = llvm.mlir.addressof @enl : !llvm.ptr
    %17 = llvm.addrspacecast %16 : !llvm.ptr to !llvm.ptr
    %18 = llvm.mlir.addressof @eae : !llvm.ptr
    %19 = llvm.addrspacecast %18 : !llvm.ptr to !llvm.ptr
    %20 = llvm.mlir.addressof @ehe : !llvm.ptr
    %21 = llvm.addrspacecast %20 : !llvm.ptr to !llvm.ptr
    %22 = llvm.mlir.addressof @ene : !llvm.ptr
    %23 = llvm.addrspacecast %22 : !llvm.ptr to !llvm.ptr

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           %[[REF:.*]] = load ptr, ptr @ial_decl_tgt_ref_ptr, align 8
    // CHECK:           store float 1.100000e+00, ptr %[[REF]], align 4
    %map_ial = omp.map.info var_ptr(%1 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ial"}
    omp.target kernel_type(generic) map_entries(%map_ial -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(1.100000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           %[[REF:.*]] = load ptr, ptr @eal_decl_tgt_ref_ptr, align 8
    // CHECK:           store float 1.200000e+00, ptr %[[REF]], align 4
    %map_eal = omp.map.info var_ptr(%13 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "eal"}
    omp.target kernel_type(generic) map_entries(%map_eal -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(1.200000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           %[[REF:.*]] = load ptr, ptr @ihl_decl_tgt_ref_ptr, align 8
    // CHECK:           store float 2.100000e+00, ptr %[[REF]], align 4
    %map_ihl = omp.map.info var_ptr(%3 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ihl"}
    omp.target kernel_type(generic) map_entries(%map_ihl -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(2.100000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           %[[REF:.*]] = load ptr, ptr @ehl_decl_tgt_ref_ptr, align 8
    // CHECK:           store float 2.200000e+00, ptr %[[REF]], align 4
    %map_ehl = omp.map.info var_ptr(%15 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ehl"}
    omp.target kernel_type(generic) map_entries(%map_ehl -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(2.200000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           %[[REF:.*]] = load ptr, ptr @inl_decl_tgt_ref_ptr, align 8
    // CHECK:           store float 3.100000e+00, ptr %[[REF]], align 4
    %map_inl = omp.map.info var_ptr(%5 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "inl"}
    omp.target kernel_type(generic) map_entries(%map_inl -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(3.100000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           %[[REF:.*]] = load ptr, ptr @enl_decl_tgt_ref_ptr, align 8
    // CHECK:           store float 3.200000e+00, ptr %[[REF]], align 4
    %map_enl = omp.map.info var_ptr(%17 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "enl"}
    omp.target kernel_type(generic) map_entries(%map_enl -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(3.200000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           store float 4.100000e+00, ptr @iae, align 4
    %map_iae = omp.map.info var_ptr(%7 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "iae"}
    omp.target kernel_type(generic) map_entries(%map_iae -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(4.100000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           store float 4.200000e+00, ptr @eae, align 4
    %map_eae = omp.map.info var_ptr(%19 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "eae"}
    omp.target kernel_type(generic) map_entries(%map_eae -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(4.200000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           store float 5.100000e+00, ptr @ihe, align 4
    %map_ihe = omp.map.info var_ptr(%9 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ihe"}
    omp.target kernel_type(generic) map_entries(%map_ihe -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(5.100000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           store float 5.200000e+00, ptr @ehe, align 4
    %map_ehe = omp.map.info var_ptr(%21 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ehe"}
    omp.target kernel_type(generic) map_entries(%map_ehe -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(5.200000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           store float 6.100000e+00, ptr @ine, align 4
    %map_ine = omp.map.info var_ptr(%11 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ine"}
    omp.target kernel_type(generic) map_entries(%map_ine -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(6.100000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    // CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_l{{[0-9]+}}(
    // CHECK:         omp.target:
    // CHECK:           store float 6.210000e+00, ptr @ene, align 4
    %map_ene = omp.map.info var_ptr(%23 : !llvm.ptr, f32) map_clauses(always, from) capture(ByRef) -> !llvm.ptr {name = "ene"}
    omp.target kernel_type(generic) map_entries(%map_ene -> %arg0 : !llvm.ptr) {
      %v = llvm.mlir.constant(6.210000e+00 : f32) : f32
      llvm.store %v, %arg0 : f32, !llvm.ptr
      omp.terminator
    }

    llvm.return
  }
}
