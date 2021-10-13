// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func @unknown_clause() {
  // expected-error@+1 {{invalid is not a valid clause for the omp.parallel operation}}
  omp.parallel invalid {
  }

  return
}

// -----

func @if_once(%n : i1) {
  // expected-error@+1 {{at most one if clause can appear on the omp.parallel operation}}
  omp.parallel if(%n : i1) if(%n : i1) {
  }

  return
}

// -----

func @num_threads_once(%n : si32) {
  // expected-error@+1 {{at most one num_threads clause can appear on the omp.parallel operation}}
  omp.parallel num_threads(%n : si32) num_threads(%n : si32) {
  }

  return
}

// -----

func @private_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one private clause can appear on the omp.parallel operation}}
  omp.parallel private(%n : memref<i32>) private(%n : memref<i32>) {
  }

  return
}

// -----

func @firstprivate_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one firstprivate clause can appear on the omp.parallel operation}}
  omp.parallel firstprivate(%n : memref<i32>) firstprivate(%n : memref<i32>) {
  }

  return
}

// -----

func @shared_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one shared clause can appear on the omp.parallel operation}}
  omp.parallel shared(%n : memref<i32>) shared(%n : memref<i32>) {
  }

  return
}

// -----

func @copyin_once(%n : memref<i32>) {
  // expected-error@+1 {{at most one copyin clause can appear on the omp.parallel operation}}
  omp.parallel copyin(%n : memref<i32>) copyin(%n : memref<i32>) {
  }

  return
}

// -----
 
func @default_once() {
  // expected-error@+1 {{at most one default clause can appear on the omp.parallel operation}}
  omp.parallel default(private) default(firstprivate) {
  }

  return
}

// -----

func @proc_bind_once() {
  // expected-error@+1 {{at most one proc_bind clause can appear on the omp.parallel operation}}
  omp.parallel proc_bind(close) proc_bind(spread) {
  }

  return
}

// -----

// expected-error @below {{op expects initializer region with one argument of the reduction type}}
omp.reduction.declare @add_f32 : f64
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// -----

// expected-error @below {{expects initializer region to yield a value of the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f64
  omp.yield (%0 : f64)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// -----

// expected-error @below {{expects reduction region with two arguments of the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f64, %arg1: f64):
  %1 = arith.addf %arg0, %arg1 : f64
  omp.yield (%1 : f64)
}

// -----

// expected-error @below {{expects reduction region to yield a value of the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  %2 = arith.extf %1 : f32 to f64
  omp.yield (%2 : f64)
}

// -----

// expected-error @below {{expects atomic reduction region with two arguments of the same type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg0: memref<f32>, %arg1: memref<f64>):
  omp.yield
}

// -----

// expected-error @below {{expects atomic reduction region arguments to be accumulators containing the reduction type}}
omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg0: memref<f64>, %arg1: memref<f64>):
  omp.yield
}

// -----

omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>

  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %0 : !llvm.ptr<f32>) {
    %2 = arith.constant 2.0 : f32
    // expected-error @below {{accumulator is not used by the parent}}
    omp.reduction %2, %1 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// -----

func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>

  // expected-error @below {{expected symbol reference @foo to point to a reduction declaration}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@foo -> %0 : !llvm.ptr<f32>) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %1 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// -----

omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>

  // expected-error @below {{accumulator variable used more than once}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %0 : !llvm.ptr<f32>, @add_f32 -> %0 : !llvm.ptr<f32>) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %0 : !llvm.ptr<f32>
    omp.yield
  }
  return
}

// -----

omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32>):
  %2 = llvm.load %arg3 : !llvm.ptr<f32>
  llvm.atomicrmw fadd %arg2, %2 monotonic : f32
  omp.yield
}

func @foo(%lb : index, %ub : index, %step : index, %mem : memref<1xf32>) {
  %c1 = arith.constant 1 : i32

  // expected-error @below {{expected accumulator ('memref<1xf32>') to be the same type as reduction declaration ('!llvm.ptr<f32>')}}
  omp.wsloop (%iv) : index = (%lb) to (%ub) step (%step)
  reduction(@add_f32 -> %mem : memref<1xf32>) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %mem : memref<1xf32>
    omp.yield
  }
  return
}

// -----

func @omp_critical1() -> () {
  // expected-error @below {{must specify a name unless the effect is as if no hint is specified}}
  omp.critical hint(nonspeculative) {
    omp.terminator
  }
  return
}

// -----

func @omp_critical2() -> () {
  // expected-error @below {{expected symbol reference @excl to point to a critical declaration}}
  omp.critical(@excl) hint(speculative) {
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @mutex
func @omp_critical() -> () {
  // expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
  omp.critical(@mutex) hint(uncontended, contended) {
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @mutex
func @omp_critical() -> () {
  // expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined}}
  omp.critical(@mutex) hint(nonspeculative, speculative) {
    omp.terminator
  }
  return
}

// -----

omp.critical.declare @mutex
func @omp_critica() -> () {
  // expected-error @below {{invalid_hint is not a valid hint}}
  omp.critical(@mutex) hint(invalid_hint) {
    omp.terminator
  }
}
