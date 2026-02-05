// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s


llvm.func @atomic_hint(%v : !llvm.ptr, %x : !llvm.ptr, %expr : i32) {
  // expected-warning@below {{hint clause discarded}}
  omp.atomic.capture hint(uncontended) {
    omp.atomic.read %x = %v : !llvm.ptr, !llvm.ptr, i32
    omp.atomic.write %v = %expr : !llvm.ptr, i32
  }

  // expected-warning@below {{hint clause discarded}}
  omp.atomic.read %x = %v hint(contended) : !llvm.ptr, !llvm.ptr, i32

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

llvm.func @distribute_allocate(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.distribute operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.distribute}}
  omp.distribute allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @distribute_order(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause order in omp.distribute operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.distribute}}
  omp.distribute order(concurrent) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @parallel_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.parallel operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.parallel}}
  omp.parallel allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @sections_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.sections operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.sections}}
  omp.sections allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : i32 init {
^bb0(%mold: !llvm.ptr, %private: !llvm.ptr):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.store %c0, %private : i32, !llvm.ptr
  omp.yield(%private: !llvm.ptr)
}
llvm.func @sections_private(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause privatization in omp.sections operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.sections}}
  omp.sections private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
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
llvm.func @scan_reduction(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause reduction with modifier in omp.wsloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop reduction(mod:inscan, @add_f32 %x -> %prv : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.scan inclusive(%prv : !llvm.ptr)
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @single_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.single operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.single}}
  omp.single allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : i32 init {
^bb0(%mold: !llvm.ptr, %private: !llvm.ptr):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.store %c0, %private : i32, !llvm.ptr
  omp.yield(%private: !llvm.ptr)
}
llvm.func @single_private(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause privatization in omp.single operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.single}}
  omp.single private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.target operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
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
  // expected-error@below {{not yet implemented: Unhandled clause in_reduction in omp.target operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target in_reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_enter_data_depend(%x: !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause depend in omp.target_enter_data operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target_enter_data}}
  omp.target_enter_data depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_exit_data_depend(%x: !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause depend in omp.target_exit_data operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target_exit_data}}
  omp.target_exit_data depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @target_update_depend(%x: !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause depend in omp.target_update operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target_update}}
  omp.target_update depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @task_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.task operation}}
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
  // expected-error@below {{not yet implemented: Unhandled clause in_reduction in omp.task operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task in_reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @taskgroup_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.taskgroup operation}}
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
  // expected-error@below {{not yet implemented: Unhandled clause task_reduction in omp.taskgroup operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskgroup}}
  omp.taskgroup task_reduction(@add_f32 %x -> %prv : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
// -----

llvm.func @taskloop_allocate(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----
 omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  }combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }

llvm.func @taskloop_inreduction(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause in_reduction in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop in_reduction(@add_reduction_i32 %x -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----
 omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  }combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }

llvm.func @taskloop_reduction(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause reduction in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop reduction(@add_reduction_i32 %x -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskwait_depend(%x: !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause depend in omp.taskwait operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskwait}}
  omp.taskwait depend(taskdependin -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @taskwait_nowait() {
  // expected-error@below {{not yet implemented: Unhandled clause nowait in omp.taskwait operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskwait}}
  omp.taskwait nowait {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @teams_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.teams operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

omp.private {type = private} @x.privatizer : i32 init {
^bb0(%mold: !llvm.ptr, %private: !llvm.ptr):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.store %c0, %private : i32, !llvm.ptr
  omp.yield(%private: !llvm.ptr)
}
llvm.func @teams_private(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause privatization in omp.teams operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @teams_num_teams_multi_dim(%lb : i32, %ub : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause num_teams with multi-dimensional values in omp.teams operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams num_teams(to %ub, %ub, %ub : i32, i32, i32) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @parallel_num_threads_multi_dim(%lb : i32, %ub : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause num_threads with multi-dimensional values in omp.parallel operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.parallel}}
  omp.parallel num_threads(%lb, %ub : i32, i32) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @teams_thread_limit_multi_dim(%lb : i32, %ub : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause thread_limit with multi-dimensional values in omp.teams operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams thread_limit(%lb, %ub : i32, i32) {
    omp.terminator
  }
  llvm.return
}

// -----

llvm.func @wsloop_allocate(%lb : i32, %ub : i32, %step : i32, %x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.wsloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----
llvm.func @wsloop_order(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause order in omp.wsloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.wsloop}}
  omp.wsloop order(concurrent) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----
llvm.func @task_affinity(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause affinity in omp.task operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.task}}
  omp.task affinity(%x : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
