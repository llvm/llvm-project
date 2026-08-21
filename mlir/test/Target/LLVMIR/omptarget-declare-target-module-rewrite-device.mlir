// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// This test verifies the module-scope declare target global use rewrite
// mechanism for a `declare target link` variable when compiling for device.
//
// The declare target global is used both directly inside of a target region
// and indirectly inside of a declare target function that is invoked from
// within that target region. Both uses of the original global should be
// rewritten to load from the generated reference pointer.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK-DAG: @_QMtest_0Esp_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Esp() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define {{.*}} @_QMtest_0Puse_global
  // CHECK: %[[REF:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK: store i32 2, ptr %[[REF]], align 4
  llvm.func @_QMtest_0Puse_global() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %1, %0 : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @_QQmain() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr

    // CHECK-DAG:   omp.target:
    // CHECK-DAG: %[[V:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 1, ptr %[[V]], align 4
    // CHECK-DAG: call void @_QMtest_0Puse_global()
    %map = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%map -> %arg0 : !llvm.ptr) {
      %1 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %1, %arg0 : i32, !llvm.ptr
      llvm.call @_QMtest_0Puse_global() : () -> ()
      omp.terminator
    }

    llvm.return
  }
}

// -----

// This test verifies the declare target global use rewrite mechanism when
// the original global is consumed by a PHI node. Making sure we rewrite this
// case correctly.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK-DAG: @_QMtest_0Esp_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Esp() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define hidden void @_QMtest_0Puse_global
  // CHECK: %[[LOAD0:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK: %[[LOAD1:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK: br i1 %{{.*}}, label %[[BB_A:.*]], label %[[BB_B:.*]]
  // CHECK: [[BB_A]]:
  // CHECK: %[[PHI_A:.*]] = phi ptr [ %[[LOAD1]], %{{.*}} ]
  // CHECK: br label %[[MERGE:.*]]
  // CHECK: [[BB_B]]:
  // CHECK: %[[PHI_B:.*]] = phi ptr [ %[[LOAD0]], %{{.*}} ]
  // CHECK: br label %[[MERGE]]
  // CHECK: [[MERGE]]:
  // CHECK: %[[PHI:.*]] = phi ptr [ %[[PHI_B]], %[[BB_B]] ], [ %[[PHI_A]], %[[BB_A]] ]
  // CHECK: store i32 2, ptr %[[PHI]], align 4
  llvm.func @_QMtest_0Puse_global(%cond : i1) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr
    llvm.cond_br %cond, ^bb1(%0 : !llvm.ptr), ^bb2(%0 : !llvm.ptr)
  ^bb1(%arg1 : !llvm.ptr):
    llvm.br ^bb3(%arg1 : !llvm.ptr)
  ^bb2(%arg2 : !llvm.ptr):
    llvm.br ^bb3(%arg2 : !llvm.ptr)
  ^bb3(%arg3 : !llvm.ptr):
    %1 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %1, %arg3 : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// A more complicated exercise of the declare target global use rewrite when
// the original global is consumed by multiple PHI nodes, some nested, and
// additionally the original global lives in a non-default address space
// (address space 2)

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK-DAG: @_QMtest_0Esp_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Esp() {addr_space = 2 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define hidden void @_QMtest_0Puse_global_nested
  //
  // CHECK: %[[L0:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK: br i1 %{{.*}}, label %[[A:.*]], label %[[B:.*]]
  // CHECK: [[A]]:
  // CHECK: %[[PA:.*]] = phi ptr [ %[[L0]], %[[ENTRY:.*]] ]
  // CHECK: br label %[[M1:.*]]
  // CHECK: [[B]]:
  // CHECK: %[[PB:.*]] = phi ptr [ %[[L0]], %[[ENTRY]] ]
  // CHECK: br label %[[M1]]
  // CHECK: [[M1]]:
  // CHECK: %[[PHI1:.*]] = phi ptr [ %[[PB]], %[[B]] ], [ %[[PA]], %[[A]] ]
  //
  // CHECK: %[[L1:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK: br i1 %{{.*}}, label %[[C:.*]], label %[[D:.*]]
  // CHECK: [[C]]:
  // CHECK: %[[PC:.*]] = phi ptr [ %[[PHI1]], %[[M1]] ]
  // CHECK: br label %[[M2:.*]]
  // CHECK: [[D]]:
  // CHECK: %[[PD:.*]] = phi ptr [ %[[L1]], %[[M1]] ]
  // CHECK: br label %[[M2]]
  // CHECK: [[M2]]:
  // CHECK: %[[PHI2:.*]] = phi ptr [ %[[PD]], %[[D]] ], [ %[[PC]], %[[C]] ]
  // CHECK: store i32 3, ptr %[[PHI2]], align 4
  llvm.func @_QMtest_0Puse_global_nested(%cond1 : i1, %cond2 : i1) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %g = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr<2>
    %gc = llvm.addrspacecast %g : !llvm.ptr<2> to !llvm.ptr
    llvm.cond_br %cond1, ^bb1(%gc : !llvm.ptr), ^bb2(%gc : !llvm.ptr)
  ^bb1(%a : !llvm.ptr):
    llvm.br ^merge1(%a : !llvm.ptr)
  ^bb2(%b : !llvm.ptr):
    llvm.br ^merge1(%b : !llvm.ptr)
  ^merge1(%m1 : !llvm.ptr):
    %g2 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr<2>
    %g2c = llvm.addrspacecast %g2 : !llvm.ptr<2> to !llvm.ptr
    llvm.cond_br %cond2, ^bb3(%m1 : !llvm.ptr), ^bb4(%g2c : !llvm.ptr)
  ^bb3(%c : !llvm.ptr):
    llvm.br ^merge2(%c : !llvm.ptr)
  ^bb4(%d : !llvm.ptr):
    llvm.br ^merge2(%d : !llvm.ptr)
  ^merge2(%m2 : !llvm.ptr):
    %v = llvm.mlir.constant(3 : i32) : i32
    llvm.store %v, %m2 : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// This test verifies that the module-scope declare target global use rewrite
// mechanism is NOT applied for a regular `declare target to`/`enter` variable
// (i.e. without unified shared memory) when compiling for device. In this
// configuration no reference pointer global is generated, so uses of the
// original global must remain direct references to the global itself, both
// inside of the target region and inside of a declare target function invoked
// from the target region. No `_decl_tgt_ref_ptr` global should be created and
// no load-from-reference-pointer should be emitted.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK-NOT: @_QMtest_0Evar_to_decl_tgt_ref_ptr
  // CHECK-NOT: @_QMtest_0Evar_enter_decl_tgt_ref_ptr
  // CHECK-DAG: @_QMtest_0Evar_to = global i32
  llvm.mlir.global external @_QMtest_0Evar_to() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Evar_enter = global i32
  llvm.mlir.global external @_QMtest_0Evar_enter() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
    %0 = llvm.mlir.constant(2 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define {{.*}} @_QMtest_0Puse_global
  // CHECK-NOT: load ptr, ptr @_QMtest_0Evar_to_decl_tgt_ref_ptr
  // CHECK-NOT: load ptr, ptr @_QMtest_0Evar_enter_decl_tgt_ref_ptr
  // CHECK-DAG: store i32 100, ptr @_QMtest_0Evar_to, align 4
  // CHECK-DAG: store i32 200, ptr @_QMtest_0Evar_enter, align 4
  llvm.func @_QMtest_0Puse_global() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter : !llvm.ptr
    %c100 = llvm.mlir.constant(100 : i32) : i32
    %c200 = llvm.mlir.constant(200 : i32) : i32
    llvm.store %c100, %0 : i32, !llvm.ptr
    llvm.store %c200, %1 : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @test_declare_target() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter : !llvm.ptr
    // CHECK-DAG: store i32 10, ptr @_QMtest_0Evar_to, align 4
    // CHECK-DAG: store i32 20, ptr @_QMtest_0Evar_enter, align 4
    // CHECK-DAG: call void @_QMtest_0Puse_global()
    %map0 = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    %map1 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%map0 -> %arg0, %map1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %c10 = llvm.mlir.constant(10 : i32) : i32
      %c20 = llvm.mlir.constant(20 : i32) : i32
      llvm.store %c10, %arg0 : i32, !llvm.ptr
      llvm.store %c20, %arg1 : i32, !llvm.ptr
      llvm.call @_QMtest_0Puse_global() : () -> ()
      omp.terminator
    }
    llvm.return
  }
}

// -----

// This test verifies the module-scope declare target global use rewrite
// mechanism for `declare target to` and `declare target enter` variables when
// unified shared memory is required and compiling for device. In this
// configuration a reference pointer is generated for the to/enter variables
// (as with link), so uses of the original global must be rewritten to load
// from the reference pointer at module scope.
//
// As with the link test, the globals are used both directly inside of a target
// region and indirectly inside of a declare target function invoked from that
// region, and both use-sites must be rewritten.

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

  // CHECK-LABEL: define {{.*}} @_QMtest_0Puse_global
  // CHECK-DAG: %[[TO_REF:.*]] = load ptr, ptr @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr, align 8
  // CHECK-DAG: store i32 100, ptr %[[TO_REF]], align 4
  // CHECK-DAG: %[[ENTER_REF:.*]] = load ptr, ptr @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr, align 8
  // CHECK-DAG: store i32 200, ptr %[[ENTER_REF]], align 4
  llvm.func @_QMtest_0Puse_global() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to_usm : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter_usm : !llvm.ptr
    %c100 = llvm.mlir.constant(100 : i32) : i32
    %c200 = llvm.mlir.constant(200 : i32) : i32
    llvm.store %c100, %0 : i32, !llvm.ptr
    llvm.store %c200, %1 : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @test_usm_declare_target() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to_usm : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter_usm : !llvm.ptr
    // CHECK-DAG: %[[TO_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 10, ptr %[[TO_VAR]], align 4
    // CHECK-DAG: %[[ENTER_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 20, ptr %[[ENTER_VAR]], align 4
    // CHECK-DAG: call void @_QMtest_0Puse_global()
    %map0 = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    %map1 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%map0 -> %arg0, %map1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %c10 = llvm.mlir.constant(10 : i32) : i32
      %c20 = llvm.mlir.constant(20 : i32) : i32
      llvm.store %c10, %arg0 : i32, !llvm.ptr
      llvm.store %c20, %arg1 : i32, !llvm.ptr
      llvm.call @_QMtest_0Puse_global() : () -> ()
      omp.terminator
    }
    llvm.return
  }
}
