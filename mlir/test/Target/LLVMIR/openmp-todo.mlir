// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s


llvm.func @atomic_hint(%v : !llvm.ptr, %x : !llvm.ptr, %expr : i32) {
  // expected-warning@below {{hint clause discarded}}
  omp.atomic.capture hint(uncontended) {
    omp.atomic.read %x = %v : !llvm.ptr, i32
    omp.atomic.write %v = %expr : !llvm.ptr, i32
  }

  // expected-warning@below {{hint clause discarded}}
  omp.atomic.read %x = %v hint(contended) : !llvm.ptr, i32

  // expected-warning@below {{hint clause discarded}}
  omp.atomic.write %v = %expr hint(nonspeculative) : !llvm.ptr, i32

  // expected-warning@below {{hint clause discarded}}
  omp.atomic.update hint(speculative) %x : !llvm.ptr {
  ^bb0(%arg0: i32):
    %result = llvm.add %arg0, %expr : i32
    omp.yield(%result : i32)
  }

  llvm.return
}

// -----

llvm.func @cancel() {
  // expected-error@below {{LLVM Translation failed for operation: omp.parallel}}
  omp.parallel {
    // expected-error@below {{unsupported OpenMP operation: omp.cancel}}
    // expected-error@below {{LLVM Translation failed for operation: omp.cancel}}
    omp.cancel cancellation_construct_type(parallel)
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @cancellation_point() {
  // expected-error@below {{LLVM Translation failed for operation: omp.parallel}}
  omp.parallel {
    // expected-error@below {{unsupported OpenMP operation: omp.cancellation_point}}
    // expected-error@below {{LLVM Translation failed for operation: omp.cancellation_point}}
    omp.cancellation_point cancellation_construct_type(parallel)
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @do_simd(%lb : i32, %ub : i32, %step : i32) {
  omp.wsloop {
    // expected-warning@below {{simd information on composite construct discarded}}
    omp.simd {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    } {omp.composite}
  } {omp.composite}
  llvm.return
}

// -----

llvm.func @distribute(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{unsupported OpenMP operation: omp.distribute}}
  // expected-error@below {{LLVM Translation failed for operation: omp.distribute}}
  omp.distribute {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @ordered_region_par_level_simd() {
  // expected-error@below {{parallelization-level clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.ordered.region}}
  omp.ordered.region par_level_simd {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @parallel_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.parallel}}
  omp.parallel allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @sections_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.sections}}
  omp.sections allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @sections_private(%x : !llvm.ptr) {
  // expected-error@below {{privatization clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.sections}}
  omp.sections private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @simd_aligned(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{aligned clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.simd}}
  omp.simd aligned(%x : !llvm.ptr -> 32) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @simd_linear(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{linear clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.simd}}
  omp.simd linear(%x = %step : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @simd_nontemporal(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{nontemporal clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.simd}}
  omp.simd nontemporal(%x : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @simd_private(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{privatization clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.simd}}
  omp.simd private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
llvm.func @simd_reduction(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{reduction clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.simd}}
  omp.simd reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @single_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.single}}
  omp.single allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @single_private(%x : !llvm.ptr) {
  // expected-error@below {{privatization clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.single}}
  omp.single private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_device(%x : i32) {
  // expected-error@below {{device clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target device(%x : i32) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_has_device_addr(%x : !llvm.ptr) {
  // expected-error@below {{has_device_addr clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target has_device_addr(%x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_if(%x : i1) {
  // expected-error@below {{if clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target if(%x) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
llvm.func @target_in_reduction(%x : !llvm.ptr) {
  // expected-error@below {{in_reduction clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target in_reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_is_device_ptr(%x : !llvm.ptr) {
  // expected-error@below {{is_device_ptr clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target is_device_ptr(%x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = firstprivate} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}
llvm.func @target_firstprivate(%x : !llvm.ptr) {
  // expected-error@below {{firstprivate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield
}
llvm.func @target_struct_privatization(%x : !llvm.ptr) {
  // expected-error@below {{privatization of structures not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_thread_limit(%x : i32) {
  // expected-error@below {{thread_limit clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target thread_limit(%x : i32) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_enter_data_depend(%x: !llvm.ptr) {
  // expected-error@below {{depend clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target_enter_data}}
  omp.target_enter_data depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_exit_data_depend(%x: !llvm.ptr) {
  // expected-error@below {{depend clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target_exit_data}}
  omp.target_exit_data depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_update_depend(%x: !llvm.ptr) {
  // expected-error@below {{depend clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target_update}}
  omp.target_update depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @task_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
llvm.func @task_in_reduction(%x : !llvm.ptr) {
  // expected-error@below {{in_reduction clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task in_reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @task_priority(%x : i32) {
  // expected-error@below {{priority clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task priority(%x : i32) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @task_private(%x : !llvm.ptr) {
  // expected-error@below {{privatization clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @task_untied() {
  // expected-error@below {{untied clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task untied {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @taskgroup_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskgroup}}
  omp.taskgroup allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
llvm.func @taskgroup_task_reduction(%x : !llvm.ptr) {
  // expected-error@below {{task_reduction clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskgroup}}
  omp.taskgroup task_reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @taskloop(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{unsupported OpenMP operation: omp.taskloop}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskwait_depend(%x: !llvm.ptr) {
  // expected-error@below {{depend clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskwait}}
  omp.taskwait depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @taskwait_nowait() {
  // expected-error@below {{nowait clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskwait}}
  omp.taskwait nowait {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @teams_allocate(%x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @teams_private(%x : !llvm.ptr) {
  // expected-error@below {{privatization clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
llvm.func @teams_reduction(%x : !llvm.ptr) {
  // expected-error@below {{reduction clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @wsloop_allocate(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{allocate clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @wsloop_linear(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{linear clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop linear(%x = %step : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @wsloop_order(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{order clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop order(concurrent) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @wsloop_private(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{privatization clause not yet supported}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}
