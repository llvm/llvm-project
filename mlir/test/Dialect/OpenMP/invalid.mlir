// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @unknown_clause() {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel invalid {
  }

  return
}

// -----

func.func @if_once(%n : i1) {
  // expected-error@+1 {{`if` clause can appear at most once in the expansion of the oilist directive}}
  omp.parallel if(%n : i1) if(%n : i1) {
  }

  return
}

// -----

func.func @num_threads_once(%n : si32) {
  // expected-error@+1 {{`num_threads` clause can appear at most once in the expansion of the oilist directive}}
  omp.parallel num_threads(%n : si32) num_threads(%n : si32) {
  }

  return
}

// -----

func.func @nowait_not_allowed(%n : memref<i32>) {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel nowait {}
  return
}

// -----

func.func @linear_not_allowed(%data_var : memref<i32>, %linear_var : i32) {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel linear(%data_var = %linear_var : memref<i32>)  {}
  return
}

// -----

func.func @schedule_not_allowed() {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel schedule(static) {}
  return
}

// -----

func.func @collapse_not_allowed() {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel collapse(3) {}
  return
}

// -----

func.func @order_not_allowed() {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel order(concurrent) {}
  return
}

// -----

func.func @ordered_not_allowed() {
  // expected-error@+1 {{expected '{' to begin a region}}
  omp.parallel ordered(2) {}
}

// -----

func.func @proc_bind_once() {
  // expected-error@+1 {{`proc_bind` clause can appear at most once in the expansion of the oilist directive}}
  omp.parallel proc_bind(close) proc_bind(spread) {
  }

  return
}

// -----

func.func @inclusive_not_a_clause(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{expected 'for'}}
  omp.wsloop nowait inclusive
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
}

// -----

func.func @order_value(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.wsloop order(default)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
}

// -----

func.func @if_not_allowed(%lb : index, %ub : index, %step : index, %bool_var : i1) {
  // expected-error @below {{expected 'for'}}
  omp.wsloop if(%bool_var: i1)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
}

// -----

func.func @num_threads_not_allowed(%lb : index, %ub : index, %step : index, %int_var : i32) {
  // expected-error @below {{expected 'for'}}
  omp.wsloop num_threads(%int_var: i32)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
}

// -----

func.func @proc_bind_not_allowed(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{expected 'for'}}
  omp.wsloop proc_bind(close)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
}

// -----

llvm.func @test_omp_wsloop_dynamic_bad_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{unknown modifier type: ginandtonic}}
  omp.wsloop schedule(dynamic, ginandtonic)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_many_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{unexpected modifier(s)}}
  omp.wsloop schedule(dynamic, monotonic, monotonic, monotonic)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{incorrect modifier order}}
  omp.wsloop schedule(dynamic, simd, monotonic)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier2(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{incorrect modifier order}}
  omp.wsloop schedule(dynamic, monotonic, monotonic)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier3(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{incorrect modifier order}}
  omp.wsloop schedule(dynamic, simd, simd)
  for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  llvm.return
}

// -----

func.func @omp_simdloop(%lb : index, %ub : index, %step : i32) -> () {
  // expected-error @below {{op failed to verify that all of {lowerBound, upperBound, step} have same type}}
  "omp.simdloop" (%lb, %ub, %step) ({
    ^bb0(%iv: index):
      omp.yield
  }) {operandSegmentSizes = array<i32: 1,1,1,0,0,0>} :
    (index, index, i32) -> ()

  return
}

// -----

func.func @omp_simdloop_pretty_aligned(%lb : index, %ub : index, %step : index,
                                       %data_var : memref<i32>) -> () {
  //  expected-error @below {{expected '->'}}
  omp.simdloop aligned(%data_var : memref<i32>)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  return
}

// -----

func.func @omp_simdloop_aligned_mismatch(%arg0 : index, %arg1 : index,
                                         %arg2 : index, %arg3 : memref<i32>,
                                         %arg4 : memref<i32>) -> () {
  //  expected-error @below {{op expected as many alignment values as aligned variables}}
  "omp.simdloop"(%arg0, %arg1, %arg2, %arg3, %arg4) ({
    ^bb0(%arg5: index):
      "omp.yield"() : () -> ()
  }) {alignment_values = [128],
      operandSegmentSizes = array<i32: 1, 1, 1, 2, 0, 0>} : (index, index, index, memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simdloop_aligned_negative(%arg0 : index, %arg1 : index,
                                         %arg2 : index, %arg3 : memref<i32>,
                                         %arg4 : memref<i32>) -> () {
  //  expected-error @below {{op alignment should be greater than 0}}
  "omp.simdloop"(%arg0, %arg1, %arg2, %arg3, %arg4) ({
    ^bb0(%arg5: index):
      "omp.yield"() : () -> ()
  }) {alignment_values = [-1, 128], operandSegmentSizes = array<i32: 1, 1, 1,2, 0, 0>} : (index, index, index, memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simdloop_unexpected_alignment(%arg0 : index, %arg1 : index,
                                             %arg2 : index, %arg3 : memref<i32>,
                                             %arg4 : memref<i32>) -> () {
  //  expected-error @below {{unexpected alignment values attribute}}
  "omp.simdloop"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg5: index):
      "omp.yield"() : () -> ()
  }) {alignment_values = [1, 128], operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : (index, index, index) -> ()
  return
}

// -----

func.func @omp_simdloop_aligned_float(%arg0 : index, %arg1 : index,
                                      %arg2 : index, %arg3 : memref<i32>,
                                      %arg4 : memref<i32>) -> () {
  //  expected-error @below {{failed to satisfy constraint: 64-bit integer array attribute}}
  "omp.simdloop"(%arg0, %arg1, %arg2, %arg3, %arg4) ({
    ^bb0(%arg5: index):
      "omp.yield"() : () -> ()
  }) {alignment_values = [1.5, 128], operandSegmentSizes = array<i32: 1, 1, 1,2, 0, 0>} : (index, index, index, memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simdloop_aligned_the_same_var(%arg0 : index, %arg1 : index,
                                             %arg2 : index, %arg3 : memref<i32>,
                                             %arg4 : memref<i32>) -> () {
  //  expected-error @below {{aligned variable used more than once}}
  "omp.simdloop"(%arg0, %arg1, %arg2, %arg3, %arg3) ({
    ^bb0(%arg5: index):
      "omp.yield"() : () -> ()
  }) {alignment_values = [1, 128], operandSegmentSizes = array<i32: 1, 1, 1,2, 0, 0>} : (index, index, index, memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simdloop_nontemporal_the_same_var(%arg0 : index,
                                                 %arg1 : index,
                                                 %arg2 : index,
                                                 %arg3 : memref<i32>) -> () {
  //  expected-error @below {{nontemporal variable used more than once}}
  "omp.simdloop"(%arg0, %arg1, %arg2, %arg3, %arg3) ({
    ^bb0(%arg5: index):
      "omp.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 2>} : (index, index, index, memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simdloop_order_value(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.simdloop order(default) for (%iv): index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  return
}

// -----

func.func @omp_simdloop_pretty_simdlen(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{op attribute 'simdlen' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  omp.simdloop simdlen(0) for (%iv): index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  return
}

// -----

func.func @omp_simdloop_pretty_safelen(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{op attribute 'safelen' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  omp.simdloop safelen(0) for (%iv): index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
  return
}

// -----

func.func @omp_simdloop_pretty_simdlen_safelen(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{'omp.simdloop' op simdlen clause and safelen clause are both present, but the simdlen value is not less than or equal to safelen value}}
  omp.simdloop simdlen(2) safelen(1) for (%iv): index = (%lb) to (%ub) step (%step) {
    omp.yield
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

func.func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr

  // expected-error @below {{expected symbol reference @foo to point to a reduction declaration}}
  omp.wsloop reduction(@foo %0 -> %prv : !llvm.ptr)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %1 : f32, !llvm.ptr
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

func.func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr

  // expected-error @below {{accumulator variable used more than once}}
  omp.wsloop reduction(@add_f32 %0 -> %prv : !llvm.ptr, @add_f32 %0 -> %prv1 : !llvm.ptr)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %0 : f32, !llvm.ptr
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

func.func @foo(%lb : index, %ub : index, %step : index, %mem : memref<1xf32>) {
  %c1 = arith.constant 1 : i32

  // expected-error @below {{expected accumulator ('memref<1xf32>') to be the same type as reduction declaration ('!llvm.ptr')}}
  omp.wsloop reduction(@add_f32 %mem -> %prv : memref<1xf32>)
  for (%iv) : index = (%lb) to (%ub) step (%step) {
    %2 = arith.constant 2.0 : f32
    omp.reduction %2, %mem : f32, memref<1xf32>
    omp.yield
  }
  return
}

// -----

func.func @omp_critical2() -> () {
  // expected-error @below {{expected symbol reference @excl to point to a critical declaration}}
  omp.critical(@excl) {
    omp.terminator
  }
  return
}

// -----

// expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
omp.critical.declare @mutex hint(uncontended, contended)

// -----

// expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined}}
omp.critical.declare @mutex hint(nonspeculative, speculative)

// -----

// expected-error @below {{invalid_hint is not a valid hint}}
omp.critical.declare @mutex hint(invalid_hint)

// -----

func.func @omp_ordered1(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop ordered(1)
  for (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{ordered region must be closely nested inside a worksharing-loop region with an ordered clause without parameter present}}
    omp.ordered_region {
      omp.terminator
    }
    omp.yield
  }
  return
}

// -----

func.func @omp_ordered2(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop for (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{ordered region must be closely nested inside a worksharing-loop region with an ordered clause without parameter present}}
    omp.ordered_region {
      omp.terminator
    }
    omp.yield
  }
  return
}

// -----

func.func @omp_ordered3(%vec0 : i64) -> () {
  // expected-error @below {{ordered depend directive must be closely nested inside a worksharing-loop with ordered clause with parameter present}}
  omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {num_loops_val = 1 : i64}
  return
}

// -----

func.func @omp_ordered4(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64) -> () {
  omp.wsloop ordered(0)
  for (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{ordered depend directive must be closely nested inside a worksharing-loop with ordered clause with parameter present}}
    omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {num_loops_val = 1 : i64}

    omp.yield
  }
  return
}
// -----

func.func @omp_ordered5(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64, %vec1 : i64) -> () {
  omp.wsloop ordered(1)
  for (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{number of variables in depend clause does not match number of iteration variables in the doacross loop}}
    omp.ordered depend_type(dependsource) depend_vec(%vec0, %vec1 : i64, i64) {num_loops_val = 2 : i64}

    omp.yield
  }
  return
}

// -----

func.func @omp_atomic_read1(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined.}}
  omp.atomic.read %v = %x hint(speculative, nonspeculative) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read2(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{invalid clause value: 'xyz'}}
  omp.atomic.read %v = %x memory_order(xyz) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read3(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{memory-order must not be acq_rel or release for atomic reads}}
  omp.atomic.read %v = %x memory_order(acq_rel) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read4(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{memory-order must not be acq_rel or release for atomic reads}}
  omp.atomic.read %v = %x memory_order(release) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read5(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{`memory_order` clause can appear at most once in the expansion of the oilist directive}}
  omp.atomic.read %v = %x memory_order(acquire) memory_order(relaxed) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read6(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{`hint` clause can appear at most once in the expansion of the oilist directive}}
  omp.atomic.read %v =  %x hint(speculative) hint(contended) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read6(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{read and write must not be to the same location for atomic reads}}
  omp.atomic.read %x =  %x hint(speculative) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write1(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
  omp.atomic.write  %addr = %val hint(contended, uncontended) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write2(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic writes}}
  omp.atomic.write  %addr = %val memory_order(acq_rel) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write3(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic writes}}
  omp.atomic.write  %addr = %val memory_order(acquire) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write4(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{`memory_order` clause can appear at most once in the expansion of the oilist directive}}
  omp.atomic.write  %addr = %val memory_order(release) memory_order(seq_cst) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write5(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{`hint` clause can appear at most once in the expansion of the oilist directive}}
  omp.atomic.write  %addr = %val hint(contended) hint(speculative) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write6(%addr : memref<i32>, %val : i32) {
  // expected-error @below {{invalid clause value: 'xyz'}}
  omp.atomic.write  %addr = %val memory_order(xyz) : memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_write(%addr : memref<memref<i32>>, %val : i32) {
  // expected-error @below {{address must dereference to value type}}
  omp.atomic.write %addr = %val : memref<memref<i32>>, i32
  return
}

// -----

func.func @omp_atomic_update1(%x: memref<i32>, %expr: f32) {
  // expected-error @below {{the type of the operand must be a pointer type whose element type is the same as that of the region argument}}
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: f32):
    %newval = llvm.fadd %xval, %expr : f32
    omp.yield (%newval : f32)
  }
  return
}

// -----

func.func @omp_atomic_update2(%x: memref<i32>, %expr: i32) {
  // expected-error @+2 {{op expects regions to end with 'omp.yield', found 'omp.terminator'}}
  // expected-note @below {{in custom textual format, the absence of terminator implies 'omp.yield'}}
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_update3(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic updates}}
  omp.atomic.update memory_order(acq_rel) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_update4(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{memory-order must not be acq_rel or acquire for atomic updates}}
  omp.atomic.update memory_order(acquire) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_update5(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{invalid kind of type specified}}
  omp.atomic.update %x : i32 {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_update6(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{only updated value must be returned}}
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval, %expr : i32, i32)
  }
  return
}

// -----

func.func @omp_atomic_update7(%x: memref<i32>, %expr: i32, %y: f32) {
  // expected-error @below {{input and yielded value must have the same type}}
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%y: f32)
  }
  return
}

// -----

func.func @omp_atomic_update8(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{the region must accept exactly one argument}}
  omp.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32, %tmp: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
  omp.atomic.update hint(uncontended, contended) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined}}
  omp.atomic.update hint(nonspeculative, speculative) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_update(%x: memref<i32>, %expr: i32) {
  // expected-error @below {{invalid_hint is not a valid hint}}
  omp.atomic.update hint(invalid_hint) %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    omp.yield (%newval : i32)
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{expected three operations in atomic.capture region}}
  omp.atomic.capture {
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield (%newval : i32)
    }
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield (%newval : i32)
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield (%newval : i32)
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield (%newval : i32)
    }
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.write %x = %expr : memref<i32>, i32
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{updated variable in atomic.update must be captured in second operation}}
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield (%newval : i32)
    }
    omp.atomic.read %v = %y : memref<i32>, i32
    omp.terminator
  }
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{captured variable in atomic.read must be updated in second operation}}
    omp.atomic.read %v = %y : memref<i32>, i32
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield (%newval : i32)
    }
    omp.terminator
  }
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{captured variable in atomic.read must be updated in second operation}}
    omp.atomic.read %v = %x : memref<i32>, i32
    omp.atomic.write %y = %expr : memref<i32>, i32
    omp.terminator
  }
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{the hints omp_sync_hint_uncontended and omp_sync_hint_contended cannot be combined}}
  omp.atomic.capture hint(contended, uncontended) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined}}
  omp.atomic.capture hint(nonspeculative, speculative) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{invalid_hint is not a valid hint}}
  omp.atomic.capture hint(invalid_hint) {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{operations inside capture region must not have hint clause}}
  omp.atomic.capture {
    omp.atomic.update hint(uncontended) %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{operations inside capture region must not have memory_order clause}}
  omp.atomic.capture {
    omp.atomic.update memory_order(seq_cst) %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x : memref<i32>, i32
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  // expected-error @below {{operations inside capture region must not have memory_order clause}}
  omp.atomic.capture {
    omp.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }
    omp.atomic.read %v = %x memory_order(seq_cst) : memref<i32>, i32
  }
  return
}

// -----

func.func @omp_teams_parent() {
  omp.parallel {
    // expected-error @below {{expected to be nested inside of omp.target or not nested in any OpenMP dialect operations}}
    omp.teams {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_teams_allocate(%data_var : memref<i32>) {
  omp.target {
    // expected-error @below {{expected equal sizes for allocate and allocator variables}}
    "omp.teams" (%data_var) ({
      omp.terminator
    }) {operandSegmentSizes = array<i32: 0,0,0,0,1,0,0>} : (memref<i32>) -> ()
    omp.terminator
  }
  return
}

// -----

func.func @omp_teams_num_teams1(%lb : i32) {
  omp.target {
    // expected-error @below {{expected num_teams upper bound to be defined if the lower bound is defined}}
    "omp.teams" (%lb) ({
      omp.terminator
    }) {operandSegmentSizes = array<i32: 1,0,0,0,0,0,0>} : (i32) -> ()
    omp.terminator
  }
  return
}

// -----

func.func @omp_teams_num_teams2(%lb : i32, %ub : i16) {
  omp.target {
    // expected-error @below {{expected num_teams upper bound and lower bound to be the same type}}
    omp.teams num_teams(%lb : i32 to %ub : i16) {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.sections" (%data_var) ({
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,1,0>} : (memref<i32>) -> ()
  return
}

// -----

func.func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.sections" (%data_var) ({
    omp.terminator
  }) {operandSegmentSizes = array<i32: 1,0,0>} : (memref<i32>) -> ()
  return
}

// -----

func.func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected omp.section op or terminator op inside region}}
  omp.sections {
    "test.payload" () : () -> ()
  }
  return
}

// -----

func.func @omp_sections(%cond : i1) {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections if(%cond) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections num_threads(10) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections proc_bind(close) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections(%data_var : memref<i32>, %linear_var : i32) {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections linear(%data_var = %linear_var : memref<i32>) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections schedule(static, none) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections collapse(3) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections ordered(2) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{expected '{' to begin a region}}
  omp.sections order(concurrent) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_sections() {
  // expected-error @below {{failed to verify constraint: region with 1 blocks}}
  omp.sections {
    omp.section {
      omp.terminator
    }
    omp.terminator
  ^bb2:
    omp.terminator
  }
  return
}

// -----

func.func @omp_single(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.single" (%data_var) ({
    omp.barrier
  }) {operandSegmentSizes = array<i32: 1,0,0>} : (memref<i32>) -> ()
  return
}

// -----

func.func @omp_single_copyprivate(%data_var : memref<i32>) -> () {
  // expected-error @below {{inconsistent number of copyPrivate vars (= 1) and functions (= 0), both must be equal}}
  "omp.single" (%data_var) ({
    omp.barrier
  }) {operandSegmentSizes = array<i32: 0,0,1>} : (memref<i32>) -> ()
  return
}

// -----

func.func @omp_single_copyprivate(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected symbol reference @copy_func to point to a copy function}}
  omp.single copyprivate(%data_var -> @copy_func : memref<i32>) {
    omp.barrier
  }
  return
}

// -----

func.func private @copy_func(memref<i32>)

func.func @omp_single_copyprivate(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected copy function @copy_func to have 2 operands}}
  omp.single copyprivate(%data_var -> @copy_func : memref<i32>) {
    omp.barrier
  }
  return
}

// -----

func.func private @copy_func(memref<i32>, memref<f32>)

func.func @omp_single_copyprivate(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected copy function @copy_func arguments to have the same type}}
  omp.single copyprivate(%data_var -> @copy_func : memref<i32>) {
    omp.barrier
  }
  return
}

// -----

func.func private @copy_func(memref<f32>, memref<f32>)

func.func @omp_single_copyprivate(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected copy function arguments' type ('memref<f32>') to be the same as copyprivate variable's type ('memref<i32>')}}
  omp.single copyprivate(%data_var -> @copy_func : memref<i32>) {
    omp.barrier
  }
  return
}

// -----

func.func @omp_task_depend(%data_var: memref<i32>) {
  // expected-error @below {{op expected as many depend values as depend variables}}
    "omp.task"(%data_var) ({
      "omp.terminator"() : () -> ()
    }) {depends = [], operandSegmentSizes = array<i32: 0, 0, 0, 0, 1, 0, 0>} : (memref<i32>) -> ()
   "func.return"() : () -> ()
}

// -----

func.func @omp_task(%ptr: !llvm.ptr) {
  // expected-error @below {{op expected symbol reference @add_f32 to point to a reduction declaration}}
  omp.task in_reduction(@add_f32 -> %ptr : !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
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

func.func @omp_task(%ptr: !llvm.ptr) {
  // expected-error @below {{op accumulator variable used more than once}}
  omp.task in_reduction(@add_f32 -> %ptr : !llvm.ptr, @add_f32 -> %ptr : !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
}

// -----

omp.reduction.declare @add_i32 : i32
init {
^bb0(%arg: i32):
  %0 = arith.constant 0 : i32
  omp.yield (%0 : i32)
}
combiner {
^bb1(%arg0: i32, %arg1: i32):
  %1 = arith.addi %arg0, %arg1 : i32
  omp.yield (%1 : i32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> i32
  llvm.atomicrmw add %arg2, %2 monotonic : !llvm.ptr, i32
  omp.yield
}

func.func @omp_task(%mem: memref<1xf32>) {
  // expected-error @below {{op expected accumulator ('memref<1xf32>') to be the same type as reduction declaration ('!llvm.ptr')}}
  omp.task in_reduction(@add_i32 -> %mem : memref<1xf32>) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel() {
  omp.sections {
    // expected-error @below {{cancel parallel must appear inside a parallel region}}
    omp.cancel cancellation_construct_type(parallel)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel1() {
  omp.parallel {
    // expected-error @below {{cancel sections must appear inside a sections region}}
    omp.cancel cancellation_construct_type(sections)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel2() {
  omp.sections {
    // expected-error @below {{cancel loop must appear inside a worksharing-loop region}}
    omp.cancel cancellation_construct_type(loop)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel3(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop nowait
    for (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{A worksharing construct that is canceled must not have a nowait clause}}
    omp.cancel cancellation_construct_type(loop)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel4(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop ordered(1)
    for (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
    // expected-error @below {{A worksharing construct that is canceled must not have an ordered clause}}
    omp.cancel cancellation_construct_type(loop)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel5() -> () {
  omp.sections nowait {
    omp.section {
      // expected-error @below {{A sections construct that is canceled must not have a nowait clause}}
      omp.cancel cancellation_construct_type(sections)
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancellationpoint() {
  omp.sections {
    // expected-error @below {{cancellation point parallel must appear inside a parallel region}}
    omp.cancellationpoint cancellation_construct_type(parallel)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancellationpoint1() {
  omp.parallel {
    // expected-error @below {{cancellation point sections must appear inside a sections region}}
    omp.cancellationpoint cancellation_construct_type(sections)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancellationpoint2() {
  omp.sections {
    // expected-error @below {{cancellation point loop must appear inside a worksharing-loop region}}
    omp.cancellationpoint cancellation_construct_type(loop)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testmemref = "test.memref"() : () -> (memref<i32>)
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.taskloop"(%lb, %ub, %ub, %lb, %step, %step, %testmemref) ({
  ^bb0(%arg3: i32, %arg4: i32):
    "omp.terminator"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0>} : (i32, i32, i32, i32, i32, i32, memref<i32>) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%lb, %ub, %ub, %lb, %step, %step, %testf32, %testf32_2) ({
  ^bb0(%arg3: i32, %arg4: i32):
    "omp.terminator"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0>, reductions = [@add_f32]} : (i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%lb, %ub, %ub, %lb, %step, %step, %testf32) ({
  ^bb0(%arg3: i32, %arg4: i32):
    "omp.terminator"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0>, reductions = [@add_f32, @add_f32]} : (i32, i32, i32, i32, i32, i32, !llvm.ptr) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%lb, %ub, %ub, %lb, %step, %step, %testf32, %testf32_2) ({
  ^bb0(%arg3: i32, %arg4: i32):
    "omp.terminator"() : () -> ()
  }) {in_reductions = [@add_f32], operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0>} : (i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%lb, %ub, %ub, %lb, %step, %step, %testf32_2) ({
  ^bb0(%arg3: i32, %arg4: i32):
    "omp.terminator"() : () -> ()
  }) {in_reductions = [@add_f32, @add_f32], operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0>} : (i32, i32, i32, i32, i32, i32, !llvm.ptr) -> ()
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

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{if a reduction clause is present on the taskloop directive, the nogroup clause must not be specified}}
  omp.taskloop reduction(@add_f32 -> %testf32 : !llvm.ptr, @add_f32 -> %testf32_2 : !llvm.ptr) nogroup
  for (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
    omp.terminator
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

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{the same list item cannot appear in both a reduction and an in_reduction clause}}
  omp.taskloop reduction(@add_f32 -> %testf32 : !llvm.ptr) in_reduction(@add_f32 -> %testf32 : !llvm.ptr)
  for (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
    omp.terminator
  }
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testi64 = "test.i64"() : () -> (i64)
  // expected-error @below {{the grainsize clause and num_tasks clause are mutually exclusive and may not appear on the same taskloop directive}}
  omp.taskloop grain_size(%testi64: i64) num_tasks(%testi64: i64)
  for (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_threadprivate() {
  %1 = llvm.mlir.addressof @_QFsubEx : !llvm.ptr
  // expected-error @below {{op failed to verify that all of {sym_addr, tls_addr} have same type}}
  %2 = omp.threadprivate %1 : !llvm.ptr -> memref<i32>
  return
}

// -----

func.func @omp_target(%map1: memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(delete) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{to, from, tofrom and alloc map types are permitted}}
  omp.target map_entries(%mapv -> %arg0: memref<?xi32>) {
    ^bb0(%arg0: memref<?xi32>):
  }
  return
}

// -----

func.func @omp_target_data(%map1: memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)  map_clauses(delete) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{to, from, tofrom and alloc map types are permitted}}
  omp.target_data map_entries(%mapv : memref<?xi32>){}
  return
}

// -----

func.func @omp_target_data() {
  // expected-error @below {{At least one of map, useDevicePtr, or useDeviceAddr operand must be present}}
  omp.target_data {}
  return
}

// -----

func.func @omp_target_enter_data(%map1: memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(from) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{to and alloc map types are permitted}}
  omp.target_enter_data map_entries(%mapv : memref<?xi32>){}
  return
}

// -----

func.func @omp_target_enter_data_depend(%a: memref<?xi32>) {
  %0 = omp.map_info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32>
  // expected-error @below {{op expected as many depend values as depend variables}}
  omp.target_enter_data map_entries(%0: memref<?xi32> ) {operandSegmentSizes = array<i32: 0, 0, 1, 0>}
  return
}

// -----

func.func @omp_target_exit_data(%map1: memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(to) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{from, release and delete map types are permitted}}
  omp.target_exit_data map_entries(%mapv : memref<?xi32>){}
  return
}

// -----

func.func @omp_target_exit_data_depend(%a: memref<?xi32>) {
  %0 = omp.map_info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  // expected-error @below {{op expected as many depend values as depend variables}}
  omp.target_exit_data map_entries(%0: memref<?xi32> ) {operandSegmentSizes = array<i32: 0, 0, 1, 0>}
  return
}

// -----

func.func @omp_target_update_invalid_motion_type(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{at least one of to or from map types must be specified, other map types are not permitted}}
  omp.target_update_data motion_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_type_2(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(delete) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{at least one of to or from map types must be specified, other map types are not permitted}}
  omp.target_update_data motion_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(always, to) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{present, mapper and iterator map type modifiers are permitted}}
  omp.target_update_data motion_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_2(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(close, to) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{present, mapper and iterator map type modifiers are permitted}}
  omp.target_update_data motion_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_3(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(implicit, to) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{present, mapper and iterator map type modifiers are permitted}}
  omp.target_update_data motion_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_4(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(implicit, tofrom) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{either to or from map types can be specified, not both}}
  omp.target_update_data motion_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_5(%map1 : memref<?xi32>) {
  %mapv = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32> {name = ""}
  %mapv2 = omp.map_info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{either to or from map types can be specified, not both}}
  omp.target_update_data motion_entries(%mapv, %mapv2 : memref<?xi32>, memref<?xi32>)
  return
}
llvm.mlir.global internal @_QFsubEx() : i32

// -----

func.func @omp_target_update_data_depend(%a: memref<?xi32>) {
  %0 = omp.map_info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32>
  // expected-error @below {{op expected as many depend values as depend variables}}
  omp.target_update_data motion_entries(%0: memref<?xi32> ) {operandSegmentSizes = array<i32: 0, 0, 1, 0>}
  return
}

// -----

func.func @omp_target_depend(%data_var: memref<i32>) {
  // expected-error @below {{op expected as many depend values as depend variables}}
    "omp.target"(%data_var) ({
      "omp.terminator"() : () -> ()
    }) {depends = [], operandSegmentSizes = array<i32: 0, 0, 0, 1, 0>} : (memref<i32>) -> ()
   "func.return"() : () -> ()
}

// -----

func.func @omp_distribute(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.distribute"(%data_var) <{operandSegmentSizes = array<i32: 0, 1, 0>}> ({
      "omp.terminator"() : () -> ()
    }) : (memref<i32>) -> ()
}

// -----

omp.private {type = private} @x.privatizer : i32 alloc {
^bb0(%arg0: i32):
  %0 = arith.constant 0.0 : f32
  // expected-error @below {{Invalid yielded value. Expected type: 'i32', got: 'f32'}}
  omp.yield(%0 : f32)
}

// -----

omp.private {type = private} @x.privatizer : i32 alloc {
^bb0(%arg0: i32):
  // expected-error @below {{Invalid yielded value. Expected type: 'i32', got: None}}
  omp.yield
}

// -----

omp.private {type = private} @x.privatizer : i32 alloc {
^bb0(%arg0: i32):
  // expected-error @below {{expected exit block terminator to be an `omp.yield` op.}}
  omp.terminator
}

// -----

// expected-error @below {{`alloc`: expected 1 region arguments, got: 2}}
omp.private {type = private} @x.privatizer : f32 alloc {
^bb0(%arg0: f32, %arg1: f32):
  omp.yield(%arg0 : f32)
}

// -----

// expected-error @below {{`copy`: expected 2 region arguments, got: 1}}
omp.private {type = firstprivate} @x.privatizer : f32 alloc {
^bb0(%arg0: f32):
  omp.yield(%arg0 : f32)
} copy {
^bb0(%arg0: f32):
  omp.yield(%arg0 : f32)
}

// -----

// expected-error @below {{`private` clauses require only an `alloc` region.}}
omp.private {type = private} @x.privatizer : f32 alloc {
^bb0(%arg0: f32):
  omp.yield(%arg0 : f32)
} copy {
^bb0(%arg0: f32, %arg1 : f32):
  omp.yield(%arg0 : f32)
}

// -----

// expected-error @below {{`firstprivate` clauses require both `alloc` and `copy` regions.}}
omp.private {type = firstprivate} @x.privatizer : f32 alloc {
^bb0(%arg0: f32):
  omp.yield(%arg0 : f32)
}

// -----

func.func @private_type_mismatch(%arg0: index) {
// expected-error @below {{type mismatch between a private variable and its privatizer op, var type: 'index' vs. privatizer op type: '!llvm.ptr'}}
  omp.parallel private(@var1.privatizer %arg0 -> %arg2 : index) {
    omp.terminator
  }

  return
}

omp.private {type = private} @var1.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

// -----

func.func @firstprivate_type_mismatch(%arg0: index) {
  // expected-error @below {{type mismatch between a firstprivate variable and its privatizer op, var type: 'index' vs. privatizer op type: '!llvm.ptr'}}
  omp.parallel private(@var1.privatizer %arg0 -> %arg2 : index) {
    omp.terminator
  }

  return
}

omp.private {type = firstprivate} @var1.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

// -----

func.func @undefined_privatizer(%arg0: index) {
  // expected-error @below {{failed to lookup privatizer op with symbol: '@var1.privatizer'}}
  omp.parallel private(@var1.privatizer %arg0 -> %arg2 : index) {
    omp.terminator
  }

  return
}

// -----
func.func @undefined_privatizer(%arg0: !llvm.ptr) {
  // expected-error @below {{inconsistent number of private variables and privatizer op symbols, private vars: 1 vs. privatizer op symbols: 2}}
  "omp.parallel"(%arg0) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1>, privatizers = [@x.privatizer, @y.privatizer]}> ({
    ^bb0(%arg2: !llvm.ptr):
      omp.terminator
    }) : (!llvm.ptr) -> ()
  return
}
