// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Only task-modified items belong in the runtime descriptor, in either order.

// -----

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

omp.declare_reduction @mul_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(1 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.mul %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @parallel_mixed_task_first(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.parallel reduction(task @add_i32 %x -> %px, @mul_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
      omp.task in_reduction(@add_i32 %px -> %t : !llvm.ptr) {
        %one = llvm.mlir.constant(1 : i32) : i32
        %old = llvm.load %t : !llvm.ptr -> i32
        %sum = llvm.add %old, %one : i32
        llvm.store %sum, %t : i32, !llvm.ptr
        omp.terminator
      }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @parallel_mixed_task_first..omp_par.1(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: store i32 {{[01]}}, ptr %[[X:.*]],
// CHECK: store i32 {{[01]}}, ptr %[[Y:.*]],
// CHECK: %[[ITEM:.*]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK: %[[SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[ITEM]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[X]], ptr %[[SHARED]],
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.init,
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.comb,
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 0, i32 1, ptr %[[ARR]])
// CHECK-NOT: call ptr @__kmpc_taskred_modifier_init
// CHECK-DAG: %[[CAPTURE:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
// CHECK-DAG: store ptr %[[X]], ptr %[[CAPTURE]],
// CHECK-DAG: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 0)
// CHECK-NOT: call void @__kmpc_task_reduction_modifier_fini
// CHECK: call ptr @__kmpc_task_reduction_get_th_data

// -----

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

omp.declare_reduction @mul_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(1 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.mul %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @parallel_mixed_default_first(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.parallel reduction(@mul_i32 %x -> %px, task @add_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
      omp.task in_reduction(@add_i32 %py -> %t : !llvm.ptr) {
        %one = llvm.mlir.constant(1 : i32) : i32
        %old = llvm.load %t : !llvm.ptr -> i32
        %sum = llvm.add %old, %one : i32
        llvm.store %sum, %t : i32, !llvm.ptr
        omp.terminator
      }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @parallel_mixed_default_first..omp_par.1(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: store i32 {{[01]}}, ptr %[[X:.*]],
// CHECK: store i32 {{[01]}}, ptr %[[Y:.*]],
// CHECK: %[[ITEM:.*]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK: %[[SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[ITEM]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[Y]], ptr %[[SHARED]],
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.init,
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.comb,
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 0, i32 1, ptr %[[ARR]])
// CHECK-NOT: call ptr @__kmpc_taskred_modifier_init
// CHECK-DAG: %[[CAPTURE:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
// CHECK-DAG: store ptr %[[Y]], ptr %[[CAPTURE]],
// CHECK-DAG: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 0)
// CHECK-NOT: call void @__kmpc_task_reduction_modifier_fini
// CHECK: call ptr @__kmpc_task_reduction_get_th_data

// -----

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

omp.declare_reduction @mul_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(1 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.mul %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @wsloop_mixed_task_first(%x: !llvm.ptr, %y: !llvm.ptr) {
  %lb = llvm.mlir.constant(0 : i32) : i32
  %ub = llvm.mlir.constant(4 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.wsloop reduction(task @add_i32 %x -> %px, @mul_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.task in_reduction(@add_i32 %px -> %t : !llvm.ptr) {
        %one = llvm.mlir.constant(1 : i32) : i32
        %old = llvm.load %t : !llvm.ptr -> i32
        %sum = llvm.add %old, %one : i32
        llvm.store %sum, %t : i32, !llvm.ptr
        omp.terminator
      }
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @wsloop_mixed_task_first(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: store i32 {{[01]}}, ptr %[[X:.*]],
// CHECK: store i32 {{[01]}}, ptr %[[Y:.*]],
// CHECK: %[[ITEM:.*]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK: %[[SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[ITEM]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[X]], ptr %[[SHARED]],
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.init,
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.comb,
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, i32 1, ptr %[[ARR]])
// CHECK-NOT: call ptr @__kmpc_taskred_modifier_init
// CHECK-DAG: %[[CAPTURE:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
// CHECK-DAG: store ptr %[[X]], ptr %[[CAPTURE]],
// CHECK-DAG: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 1)
// CHECK-NOT: call void @__kmpc_task_reduction_modifier_fini
// CHECK: call ptr @__kmpc_task_reduction_get_th_data

// -----

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

omp.declare_reduction @mul_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(1 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.mul %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @wsloop_mixed_default_first(%x: !llvm.ptr, %y: !llvm.ptr) {
  %lb = llvm.mlir.constant(0 : i32) : i32
  %ub = llvm.mlir.constant(4 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.wsloop reduction(@mul_i32 %x -> %px, task @add_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.task in_reduction(@add_i32 %py -> %t : !llvm.ptr) {
        %one = llvm.mlir.constant(1 : i32) : i32
        %old = llvm.load %t : !llvm.ptr -> i32
        %sum = llvm.add %old, %one : i32
        llvm.store %sum, %t : i32, !llvm.ptr
        omp.terminator
      }
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @wsloop_mixed_default_first(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: store i32 {{[01]}}, ptr %[[X:.*]],
// CHECK: store i32 {{[01]}}, ptr %[[Y:.*]],
// CHECK: %[[ITEM:.*]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK: %[[SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[ITEM]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[Y]], ptr %[[SHARED]],
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.init,
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.comb,
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, i32 1, ptr %[[ARR]])
// CHECK-NOT: call ptr @__kmpc_taskred_modifier_init
// CHECK-DAG: %[[CAPTURE:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
// CHECK-DAG: store ptr %[[Y]], ptr %[[CAPTURE]],
// CHECK-DAG: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 1)
// CHECK-NOT: call void @__kmpc_task_reduction_modifier_fini
// CHECK: call ptr @__kmpc_task_reduction_get_th_data

// -----

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

omp.declare_reduction @mul_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(1 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.mul %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @sections_mixed_task_first(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.sections reduction(task @add_i32 %x -> %px, @mul_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.section {
    ^bb0(%sx: !llvm.ptr, %sy: !llvm.ptr):
      omp.task in_reduction(@add_i32 %sx -> %t : !llvm.ptr) {
        %one = llvm.mlir.constant(1 : i32) : i32
        %old = llvm.load %t : !llvm.ptr -> i32
        %sum = llvm.add %old, %one : i32
        llvm.store %sum, %t : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @sections_mixed_task_first(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: store i32 {{[01]}}, ptr %[[X:.*]],
// CHECK: store i32 {{[01]}}, ptr %[[Y:.*]],
// CHECK: %[[ITEM:.*]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK: %[[SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[ITEM]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[X]], ptr %[[SHARED]],
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.init,
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.comb,
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, i32 1, ptr %[[ARR]])
// CHECK-NOT: call ptr @__kmpc_taskred_modifier_init
// CHECK-DAG: %[[CAPTURE:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
// CHECK-DAG: store ptr %[[X]], ptr %[[CAPTURE]],
// CHECK-DAG: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 1)
// CHECK-NOT: call void @__kmpc_task_reduction_modifier_fini
// CHECK: call ptr @__kmpc_task_reduction_get_th_data

// -----

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

omp.declare_reduction @mul_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(1 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.mul %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @sections_mixed_default_first(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.sections reduction(@mul_i32 %x -> %px, task @add_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.section {
    ^bb0(%sx: !llvm.ptr, %sy: !llvm.ptr):
      omp.task in_reduction(@add_i32 %sy -> %t : !llvm.ptr) {
        %one = llvm.mlir.constant(1 : i32) : i32
        %old = llvm.load %t : !llvm.ptr -> i32
        %sum = llvm.add %old, %one : i32
        llvm.store %sum, %t : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @sections_mixed_default_first(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: store i32 {{[01]}}, ptr %[[X:.*]],
// CHECK: store i32 {{[01]}}, ptr %[[Y:.*]],
// CHECK: %[[ITEM:.*]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK: %[[SHARED:.*]] = getelementptr inbounds nuw %kmp_taskred_input_t, ptr %[[ITEM]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[Y]], ptr %[[SHARED]],
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.init,
// CHECK: store ptr @__omp_taskred_mod_add_i32.red.comb,
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, i32 1, ptr %[[ARR]])
// CHECK-NOT: call ptr @__kmpc_taskred_modifier_init
// CHECK-DAG: %[[CAPTURE:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
// CHECK-DAG: store ptr %[[Y]], ptr %[[CAPTURE]],
// CHECK-DAG: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 1)
// CHECK-NOT: call void @__kmpc_task_reduction_modifier_fini
// CHECK: call ptr @__kmpc_task_reduction_get_th_data

// -----

// A byref DEFAULT item does not impose the TASK byref restriction.

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

  omp.declare_reduction @add_reduction_i_32 : !llvm.ptr alloc {
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 : (i64) -> !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } init {
  ^bb0(%arg0: !llvm.ptr, %alloc: !llvm.ptr):
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %0, %alloc : i32, !llvm.ptr
    omp.yield(%alloc : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    %1 = llvm.load %arg1 : !llvm.ptr -> i32
    %2 = llvm.add %0, %1  : i32
    llvm.store %2, %arg0 : i32, !llvm.ptr
    omp.yield(%arg0 : !llvm.ptr)
  }


llvm.func @mixed_byref(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.parallel reduction(byref @add_reduction_i_32 %x -> %px, task @add_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @mixed_byref..omp_par(
// CHECK: %[[ARR:.*]] = alloca [1 x %kmp_taskred_input_t]
// CHECK: call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 0, i32 1, ptr %[[ARR]])
// CHECK: call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 0)
