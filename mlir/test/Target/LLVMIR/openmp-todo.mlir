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

llvm.func @scope_allocate(%x : !llvm.ptr) {
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.scope operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.scope}}
  omp.scope allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
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
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop.context}}
  // expected-error@below {{not yet implemented: Unhandled clause allocate in omp.taskloop.context operation}}
  omp.taskloop.context allocate(%x : !llvm.ptr -> %x : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
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
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop.context}}
  // expected-error@below {{not yet implemented: Unhandled clause in_reduction in omp.taskloop.context operation}}
  omp.taskloop.context in_reduction(@add_reduction_i32 %x -> %arg0 : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
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
  // expected-error@below {{LLVM Translation failed for operation: omp.taskloop.context}}
  // expected-error@below {{not yet implemented: Unhandled clause reduction in omp.taskloop.context operation}}
  omp.taskloop.context reduction(@add_reduction_i32 %x -> %arg0 : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
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

llvm.func @teams_dyn_groupprivate(%dyn_size : i32) {
  // expected-error@below {{not yet implemented: Unhandled clause dyn_groupprivate in omp.teams operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.teams}}
  omp.teams dyn_groupprivate(%dyn_size : i32) {
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

llvm.func @target_map_iterated_unsupported(%addr : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c10 = llvm.mlir.constant(10 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %map = omp.map.info var_ptr(%addr : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %it = omp.iterator(%iv: i64) = (%c0 to %c10 step %c1) {
    %m = omp.map.info var_ptr(%addr : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.yield(%m : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  // expected-error@below {{not yet implemented: Unhandled clause map/motion clause with iterator modifier in omp.target operation}}
  // expected-error@below {{LLVM Translation failed for operation: omp.target}}
  omp.target map_iterated(%it : !omp.iterated<!llvm.ptr>) map_entries(%map -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @mapper_with_iterator : !llvm.struct<"mapper_type", (i32)> {
  ^bb0(%arg: !llvm.ptr):
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c10 = llvm.mlir.constant(10 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c10 step %c1) {
      %m = omp.map.info var_ptr(%arg : !llvm.ptr, !llvm.struct<"mapper_type", (i32)>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
      omp.yield(%m : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    // expected-error@below {{not yet implemented: Unhandled clause map/motion clause with iterator modifier in omp.declare_mapper.info operation}}
    omp.declare_mapper.info map_iterated(%it : !omp.iterated<!llvm.ptr>)
  }

  llvm.func @target_data_mapper_iterator(%addr : !llvm.ptr) {
    %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"mapper_type", (i32)>) map_clauses(tofrom) capture(ByRef) mapper(@mapper_with_iterator) -> !llvm.ptr {name = ""}
    // expected-error@below {{LLVM Translation failed for operation: omp.target_data}}
    omp.target_data map_entries(%map : !llvm.ptr) {}
    llvm.return
  }
}

// -----

llvm.func @target_enter_data_map_iterator_members(%addr : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c10 = llvm.mlir.constant(10 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %it = omp.iterator(%iv: i64) = (%c0 to %c10 step %c1) {
    %field = llvm.getelementptr %addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"map_type", (i32)>
    %field_map = omp.map.info var_ptr(%field : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}
    // expected-error@below {{not yet implemented: map/motion clause with iterator modifier that expands to member maps}}
    %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"map_type", (i32)>) map_clauses(to) capture(ByRef) members(%field_map : [0] : !llvm.ptr) -> !llvm.ptr {name = ""}
    omp.yield(%map : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  // expected-error@below {{LLVM Translation failed for operation: omp.target_enter_data}}
  omp.target_enter_data map_iterated(%it : !omp.iterated<!llvm.ptr>) {}
  llvm.return
}
