// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// The `task` reduction modifier opens a task-reduction scope around the
// parallel / worksharing region. Verify that
// __kmpc_taskred_modifier_init is emitted (with the correct `is_ws` argument)
// after the reduction privates are set up, and that
// __kmpc_task_reduction_modifier_fini is emitted before the reduction combine.

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

llvm.func @parallel_task_reduction(%x: !llvm.ptr) {
  omp.parallel reduction(mod: task, @add_i32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK: %kmp_taskred_input_t = type { ptr, ptr, i64, ptr, ptr, ptr, i32 }

// On a parallel construct the modifier init uses is_ws = 0.
// CHECK-LABEL: define internal void @parallel_task_reduction..omp_par
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 0, i32 1, ptr %[[ARR]])
// CHECK:         call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 0)

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

llvm.func @wsloop_task_reduction(%x: !llvm.ptr) {
  %lb = llvm.mlir.constant(1 : i32) : i32
  %ub = llvm.mlir.constant(10 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.wsloop reduction(mod: task, @add_i32 %x -> %prv : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// On a worksharing construct the modifier init uses is_ws = 1.
// CHECK-LABEL: define void @wsloop_task_reduction(
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, i32 1, ptr %[[ARR]])
// CHECK:         call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 1)

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

llvm.func @sections_task_reduction(%x: !llvm.ptr) {
  omp.sections reduction(mod: task, @add_i32 %x -> %prv : !llvm.ptr) {
    omp.section {
    ^bb0(%arg: !llvm.ptr):
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// On a worksharing (sections) construct the modifier init uses is_ws = 1.
// CHECK-LABEL: define void @sections_task_reduction(
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 1, i32 1, ptr %[[ARR]])
// CHECK:         call void @__kmpc_task_reduction_modifier_fini(ptr @{{.+}}, i32 %{{.+}}, i32 1)

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

llvm.func @parallel_two_task_reductions(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.parallel reduction(mod: task, @add_i32 %x -> %p0, @add_i32 %y -> %p1 : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// With two task-modifier reductions the descriptor array holds two entries and
// the modifier init receives num = 2 (is_ws = 0 on the parallel construct).
// CHECK-LABEL: define internal void @parallel_two_task_reductions..omp_par
// CHECK:         %[[ARR:.+]] = alloca [2 x %kmp_taskred_input_t]
// CHECK:         call ptr @__kmpc_taskred_modifier_init(ptr @{{.+}}, i32 %{{.+}}, i32 0, i32 2, ptr %[[ARR]])

// -----

// An empty omp.sections (only a terminator, no omp.section) hits the
// empty-sections early return, so no task-reduction scope is opened: neither
// the modifier init nor the matching fini may be emitted.

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

llvm.func @empty_sections_task_reduction(%x: !llvm.ptr) {
  omp.sections reduction(mod: task, @add_i32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @empty_sections_task_reduction(
// CHECK-NOT:     @__kmpc_taskred_modifier_init
// CHECK-NOT:     @__kmpc_task_reduction_modifier_fini
// CHECK:         ret void

// -----

// A verifier-valid omp.parallel that carries reduction_mod = task but has no
// reduction variables must not open a task-reduction scope.

llvm.func @parallel_task_mod_no_reductions() {
  "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>, reduction_mod = #omp<reduction_modifier(task)>}> ({
    omp.terminator
  }) : () -> ()
  llvm.return
}

// CHECK-LABEL: define internal void @parallel_task_mod_no_reductions..omp_par
// CHECK-NOT:     @__kmpc_taskred_modifier_init
// CHECK-NOT:     @__kmpc_task_reduction_modifier_fini
// CHECK:         ret void

// -----

// A verifier-valid omp.wsloop that carries reduction_mod = task but has no
// reduction variables must not open a task-reduction scope.

llvm.func @wsloop_task_mod_no_reductions() {
  %lb = llvm.mlir.constant(1 : i32) : i32
  %ub = llvm.mlir.constant(10 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, reduction_mod = #omp<reduction_modifier(task)>}> ({
    "omp.loop_nest"(%lb, %ub, %step) <{loop_inclusive}> ({
    ^bb0(%iv: i32):
      "omp.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
  }) : () -> ()
  llvm.return
}

// CHECK-LABEL: define void @wsloop_task_mod_no_reductions(
// CHECK-NOT:     @__kmpc_taskred_modifier_init
// CHECK-NOT:     @__kmpc_task_reduction_modifier_fini
// CHECK:         ret void

// -----

// A verifier-valid omp.sections that carries reduction_mod = task but has no
// reduction variables must not open a task-reduction scope. A section body is
// present, so this exercises the reduction-count guard rather than the
// empty-sections early return tested above.

llvm.func @sections_task_mod_no_reductions() {
  "omp.sections"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0>, reduction_mod = #omp<reduction_modifier(task)>}> ({
    "omp.section"() ({
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.terminator"() : () -> ()
  }) : () -> ()
  llvm.return
}

// CHECK-LABEL: define void @sections_task_mod_no_reductions(
// CHECK-NOT:     @__kmpc_taskred_modifier_init
// CHECK-NOT:     @__kmpc_task_reduction_modifier_fini
// CHECK:         ret void
