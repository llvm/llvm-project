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

llvm.func @ordered_region_par_level_simd() {
  // expected-error@below {{not yet implemented: Unhandled clause parallelization-level in omp.ordered.region operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.ordered.region}}
  omp.ordered.region par_level_simd {
    omp.terminator
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

llvm.func @target_device(%x : i32) {
  omp.target device(%x : i32) {
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

llvm.func @taskloop_collapse(%lb : i32, %ub : i32, %step : i32, %lb1 : i32, %ub1 : i32, %step1 : i32) {
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop {
    // expected-error@below {{not yet implemented: Unhandled clause collapse in omp.loop_nest operation}}
    // expected-error@below {{LLVM Translation failed for operation: omp.loop_nest}}
    omp.loop_nest (%iv, %iv1) : i32 = (%lb, %lb1) to (%ub, %ub1) inclusive step (%step, %step1) collapse(2) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskloop_final(%lb : i32, %ub : i32, %step : i32, %true : i1) {
  // expected-error@below {{not yet implemented: Unhandled clause final in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop final(%true) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskloop_grainsize(%lb : i32, %ub : i32, %step : i32, %grainsize : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause grainsize in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop grainsize(%grainsize: i32) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskloop_if(%lb : i32, %ub : i32, %step : i32, %true : i1) {
  // expected-error@below {{not yet implemented: Unhandled clause if in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop if(%true) {
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

llvm.func @taskloop_mergeable(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause mergeable in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop mergeable {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskloop_nogroup(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause nogroup in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop nogroup {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskloop_num_tasks(%lb : i32, %ub : i32, %step : i32, %numtasks : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause num_tasks in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop num_tasks(%numtasks: i32) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @taskloop_priority(%lb : i32, %ub : i32, %step : i32, %priority : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause priority in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop priority(%priority: i32) {
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

llvm.func @taskloop_untied(%lb : i32, %ub : i32, %step : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause untied in omp.taskloop operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop}}
  omp.taskloop untied {
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
