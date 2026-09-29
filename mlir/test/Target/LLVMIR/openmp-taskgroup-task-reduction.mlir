// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Single scalar task_reduction on omp.taskgroup. Verifies that the
// kmp_taskred_input_t descriptor is allocated, populated, and handed off to
// __kmpc_taskred_init, and that init / combiner helper functions are emitted.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskgroup_task_reduction_single(%x: !llvm.ptr) {
  omp.taskgroup task_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK: %kmp_taskred_input_t = type { ptr, ptr, i64, ptr, ptr, ptr, i32 }

// CHECK-LABEL: define void @taskgroup_task_reduction_single(
// CHECK-SAME:    ptr %[[X:.+]])
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// Descriptor entry 0.
// CHECK:         %[[GEP0:.+]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK:         %[[SHAR:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 0
// CHECK:         store ptr %[[X]], ptr %[[SHAR]]
// CHECK:         %[[ORIG:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 1
// CHECK:         store ptr %[[X]], ptr %[[ORIG]]
// CHECK:         %[[SZF:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 2
// CHECK:         store i64 4, ptr %[[SZF]]
// CHECK:         %[[INITF:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 3
// CHECK:         store ptr @__omp_taskred_add_i32.red.init, ptr %[[INITF]]
// CHECK:         %[[FINIF:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 4
// CHECK:         store ptr null, ptr %[[FINIF]]
// CHECK:         %[[COMBF:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 5
// CHECK:         store ptr @__omp_taskred_add_i32.red.comb, ptr %[[COMBF]]
// CHECK:         %[[FLAGSF:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 6
// CHECK:         store i32 0, ptr %[[FLAGSF]]
// CHECK:         call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 1, ptr %[[ARR]])
// CHECK:         call void @__kmpc_end_taskgroup(

// CHECK-LABEL: define internal void @__omp_taskred_add_i32.red.init(
// CHECK-SAME:    ptr %priv, ptr %orig)
// CHECK:         load i32, ptr %orig
// CHECK:         store i32 0, ptr %priv
// CHECK:         ret void

// CHECK-LABEL: define internal void @__omp_taskred_add_i32.red.comb(
// CHECK-SAME:    ptr %lhs, ptr %rhs)
// CHECK:         %[[L:.+]] = load i32, ptr %lhs
// CHECK:         %[[R:.+]] = load i32, ptr %rhs
// CHECK:         %[[S:.+]] = add i32 %[[L]], %[[R]]
// CHECK:         store i32 %[[S]], ptr %lhs
// CHECK:         ret void

// -----

// Non-default address space reduction storage must be normalized before
// populating kmp_taskred_input_t, while callback ABI remains generic ptr.

omp.declare_reduction @add_i32_as1 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskgroup_task_reduction_single_as1(%x: !llvm.ptr<1>) {
  omp.taskgroup task_reduction(@add_i32_as1 %x -> %prv : !llvm.ptr<1>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskgroup_task_reduction_single_as1(
// CHECK-SAME:    ptr addrspace(1) %[[X:.+]])
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         %[[X_CAST:.+]] = addrspacecast ptr addrspace(1) %[[X]] to ptr
// CHECK:         %[[GEP0:.+]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK:         %[[SHAR:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 0
// CHECK:         store ptr %[[X_CAST]], ptr %[[SHAR]]
// CHECK:         %[[ORIG:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[GEP0]], i32 0, i32 1
// CHECK:         store ptr %[[X_CAST]], ptr %[[ORIG]]

// CHECK-LABEL: define internal void @__omp_taskred_add_i32_as1.red.init(
// CHECK-SAME:    ptr %priv, ptr %orig)
// CHECK:         load i32, ptr %orig
// CHECK:         store i32 0, ptr %priv

// CHECK-LABEL: define internal void @__omp_taskred_add_i32_as1.red.comb(
// CHECK-SAME:    ptr %lhs, ptr %rhs)
// CHECK:         %[[AS1_L:.+]] = load i32, ptr %lhs
// CHECK:         %[[AS1_R:.+]] = load i32, ptr %rhs
// CHECK:         %[[AS1_S:.+]] = add i32 %[[AS1_L]], %[[AS1_R]]
// CHECK:         store i32 %[[AS1_S]], ptr %lhs

// -----

// Pointer-typed task_reduction. Here the declare_reduction's reduction type is
// itself a pointer, so the initializer's mold argument is the storage pointer.
// Following the regular reduction path (mapInitializationArgs), the init helper
// must pass %orig through directly instead of emitting a spurious
// `load ptr, ptr %orig`. The combiner still loads its pointer operands, exactly
// like the by-value reduction path.

omp.declare_reduction @keep_ptr : !llvm.ptr
init {
^bb0(%mold: !llvm.ptr):
  omp.yield(%mold : !llvm.ptr)
}
combiner {
^bb0(%a: !llvm.ptr, %b: !llvm.ptr):
  omp.yield(%a : !llvm.ptr)
}

llvm.func @taskgroup_task_reduction_ptr(%x: !llvm.ptr) {
  omp.taskgroup task_reduction(@keep_ptr %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @__omp_taskred_keep_ptr.red.init(
// CHECK-SAME:    ptr %priv, ptr %orig)
// CHECK-NOT:     load ptr, ptr %orig
// CHECK:         store ptr %orig, ptr %priv
// CHECK:         ret void

// CHECK-LABEL: define internal void @__omp_taskred_keep_ptr.red.comb(
// CHECK-SAME:    ptr %lhs, ptr %rhs)
// CHECK:         %[[L:.+]] = load ptr, ptr %lhs
// CHECK:         %[[R:.+]] = load ptr, ptr %rhs
// CHECK:         store ptr %[[L]], ptr %lhs
// CHECK:         ret void

// -----

// Multiple task_reduction items on the same taskgroup.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

omp.declare_reduction @mul_i64 : i64
init {
^bb0(%arg0: i64):
  %c1 = llvm.mlir.constant(1 : i64) : i64
  omp.yield(%c1 : i64)
}
combiner {
^bb0(%arg0: i64, %arg1: i64):
  %p = llvm.mul %arg0, %arg1 : i64
  omp.yield(%p : i64)
}

llvm.func @taskgroup_task_reduction_multi(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.taskgroup task_reduction(@add_i32 %x -> %a, @mul_i64 %y -> %b : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskgroup_task_reduction_multi(
// CHECK-SAME:    ptr %[[XA:[^,)]+]], ptr %[[YA:[^,)]+]])
// CHECK:         %[[ARR2:.+]] = alloca [2 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// Descriptor entry 0: @add_i32 on %x.
// CHECK:         %[[E0:.+]] = getelementptr inbounds [2 x %kmp_taskred_input_t], ptr %[[ARR2]], i32 0, i32 0
// CHECK:         %[[E0_SHAR:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 0
// CHECK:         store ptr %[[XA]], ptr %[[E0_SHAR]]
// CHECK:         %[[E0_ORIG:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 1
// CHECK:         store ptr %[[XA]], ptr %[[E0_ORIG]]
// CHECK:         %[[E0_SZ:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 2
// CHECK:         store i64 4, ptr %[[E0_SZ]]
// CHECK:         %[[E0_INIT:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 3
// CHECK:         store ptr @__omp_taskred_add_i32.red.init, ptr %[[E0_INIT]]
// CHECK:         %[[E0_FINI:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 4
// CHECK:         store ptr null, ptr %[[E0_FINI]]
// CHECK:         %[[E0_COMB:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 5
// CHECK:         store ptr @__omp_taskred_add_i32.red.comb, ptr %[[E0_COMB]]
// CHECK:         %[[E0_FL:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E0]], i32 0, i32 6
// CHECK:         store i32 0, ptr %[[E0_FL]]
// Descriptor entry 1: @mul_i64 on %y.
// CHECK:         %[[E1:.+]] = getelementptr inbounds [2 x %kmp_taskred_input_t], ptr %[[ARR2]], i32 0, i32 1
// CHECK:         %[[E1_SHAR:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 0
// CHECK:         store ptr %[[YA]], ptr %[[E1_SHAR]]
// CHECK:         %[[E1_ORIG:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 1
// CHECK:         store ptr %[[YA]], ptr %[[E1_ORIG]]
// CHECK:         %[[E1_SZ:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 2
// CHECK:         store i64 8, ptr %[[E1_SZ]]
// CHECK:         %[[E1_INIT:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 3
// CHECK:         store ptr @__omp_taskred_mul_i64.red.init, ptr %[[E1_INIT]]
// CHECK:         %[[E1_FINI:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 4
// CHECK:         store ptr null, ptr %[[E1_FINI]]
// CHECK:         %[[E1_COMB:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 5
// CHECK:         store ptr @__omp_taskred_mul_i64.red.comb, ptr %[[E1_COMB]]
// CHECK:         %[[E1_FL:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[E1]], i32 0, i32 6
// CHECK:         store i32 0, ptr %[[E1_FL]]
// CHECK:         call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 2, ptr %[[ARR2]])

// -----

// Plain taskgroup without task_reduction must still translate (regression
// guard for the rewrite of convertOmpTaskgroupOp).

llvm.func @taskgroup_plain() {
  omp.taskgroup {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskgroup_plain()
// CHECK:         call void @__kmpc_taskgroup(
// CHECK-NOT:     call ptr @__kmpc_taskred_init(
// CHECK:         call void @__kmpc_end_taskgroup(
