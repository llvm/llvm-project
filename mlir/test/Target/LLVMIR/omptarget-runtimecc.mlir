// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module attributes {omp.is_target_device = true,  omp.is_gpu = true, omp.target_triples = ["spirv64-intel"], llvm.target_triple = "spirv64-intel"} {
// CHECK: call spir_func i32 @__kmpc_target_init
// CHECK: call spir_func void @__kmpc_target_deinit
  llvm.func @target_if_variable(%x : i1) {
    omp.target kernel_type(generic) if(%x) {
      omp.terminator
    }
    llvm.return
  }
 }

// -----

// Test that the calls to these runtime functions get the runtime calling
// convention:
//   __kmpc_taskgroup
//   __kmpc_omp_task_alloc
//   __kmpc_taskloop
//   __kmpc_end_taskgroup
//   __kmpc_aligned_alloc
//   __kmpc_task_reduction_get_th_data
//   __kmpc_get_shared_variables
//   __kmpc_alloc_shared
//   __kmpc_free_shared
//   __kmpc_taskred_init
//   __kmpc_taskred_modifier_init
//   __kmpc_task_reduction_modifier_fini

module attributes {llvm.target_triple = "spirv64-intel"} {
  omp.private {type = private} @i.private : i32

  llvm.func @taskloop() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %lb = llvm.mlir.constant(1 : i32) : i32
    %ub = llvm.mlir.constant(10 : i32) : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.taskloop.context private(@i.private %1 -> %arg0 : !llvm.ptr) {
      omp.taskloop.wrapper {
        omp.loop_nest (%arg1) : i32 = (%lb) to (%ub) inclusive step (%step) {
          llvm.store %arg1, %arg0 : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    } {omp.combined}
    llvm.return
  }
}

// CHECK-LABEL: define void @taskloop(
// CHECK:         call spir_func void @__kmpc_taskgroup(
// CHECK:         call spir_func ptr @__kmpc_omp_task_alloc(
// CHECK:         call spir_func void @__kmpc_taskloop(
// CHECK:         call spir_func void @__kmpc_end_taskgroup(

// -----

module attributes {llvm.target_triple = "spirv64-intel"} {
  omp.private {type = private} @y.private : i32

  llvm.func @allocate_aligned(%x: !llvm.ptr, %allocator: i64) {
    omp.parallel allocate(%allocator : i64 -> %x : !llvm.ptr)
        allocate_alignments([64]) allocate_private_indices([0])
        private(@y.private %x -> %x.private : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @allocate_aligned..omp_par(
// CHECK:         call spir_func ptr @__kmpc_aligned_alloc(

// -----

module attributes {llvm.target_triple = "spirv64-intel"} {
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

  llvm.func @target_inreduction(%x : !llvm.ptr) {
    %m = omp.map.info var_ptr(%x : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) in_reduction(@add_i32 %x : !llvm.ptr) map_entries(%m -> %marg : !llvm.ptr) {
      %v = llvm.load %marg : !llvm.ptr -> i32
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %s = llvm.add %v, %c1 : i32
      llvm.store %s, %marg : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_inreduction_
// CHECK:         call spir_func ptr @__kmpc_task_reduction_get_th_data(

// -----

module attributes {omp.is_target_device = true, omp.is_gpu = true,
                   omp.target_triples = ["spirv64-intel"],
                   llvm.target_triple = "spirv64-intel"} {
  llvm.func @target_generic_parallel(%p : !llvm.ptr) {
    %mi = omp.map.info var_ptr(%p : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%mi -> %a : !llvm.ptr) {
      omp.parallel {
        %c = llvm.mlir.constant(42 : i32) : i32
        llvm.store %c, %a : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_generic_parallel_{{.*}}.wrapper(
// CHECK:         call spir_func void @__kmpc_get_shared_variables(

// -----

module attributes {omp.is_target_device = true, omp.is_gpu = true,
                   omp.target_triples = ["spirv64-intel"],
                   llvm.target_triple = "spirv64-intel"} {
  omp.private {type = private} @privatizer : i32

  llvm.func @device_shared_mem(%p : !llvm.ptr) {
    %m = omp.map.info var_ptr(%p : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %cn = llvm.mlir.constant(10 : i32) : i32
      omp.distribute private(@privatizer %arg0 -> %arg1 : !llvm.ptr) {
        omp.loop_nest (%i) : i32 = (%c1) to (%cn) inclusive step (%c1) {
          llvm.store %i, %arg1 : i32, !llvm.ptr
          omp.parallel {
            %v = llvm.load %arg1 : !llvm.ptr -> i32
            omp.terminator
          }
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_device_shared_mem_
// CHECK:         call spir_func align 8 ptr @__kmpc_alloc_shared(
// CHECK:         call spir_func void @__kmpc_free_shared(

// -----

module attributes {llvm.target_triple = "spirv64-intel"} {
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

  llvm.func @task_in_reduction(%x : !llvm.ptr) {
    omp.task in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
      %v = llvm.load %prv : !llvm.ptr -> i32
      %c1 = llvm.mlir.constant(1 : i32) : i32
      %s = llvm.add %v, %c1 : i32
      llvm.store %s, %prv : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @task_in_reduction..omp_par(
// CHECK:         call spir_func ptr @__kmpc_task_reduction_get_th_data(

// -----

module attributes {llvm.target_triple = "spirv64-intel"} {
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

  llvm.func @taskloop_reduction(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
    omp.taskloop.context reduction(@add_i32 %x -> %prv : !llvm.ptr) {
      omp.taskloop.wrapper {
        omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
          %v = llvm.load %prv : !llvm.ptr -> i32
          %s = llvm.add %v, %iv : i32
          llvm.store %s, %prv : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    } {omp.combined}
    llvm.return
  }
}

// CHECK-LABEL: define void @taskloop_reduction(
// CHECK:         call spir_func void @__kmpc_taskgroup(
// CHECK:         call spir_func ptr @__kmpc_taskred_init(
// CHECK:         call spir_func void @__kmpc_end_taskgroup(
// CHECK-LABEL: define internal void @taskloop_reduction..omp_par(
// CHECK:         call spir_func ptr @__kmpc_task_reduction_get_th_data(

// -----

module attributes {llvm.target_triple = "spirv64-intel"} {
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

  llvm.func @parallel_task_reduction(%x : !llvm.ptr) {
    omp.parallel reduction(task @add_i32 %x -> %prv : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @parallel_task_reduction..omp_par(
// CHECK:         call spir_func ptr @__kmpc_taskred_modifier_init(
// CHECK:         call spir_func void @__kmpc_task_reduction_modifier_fini(
