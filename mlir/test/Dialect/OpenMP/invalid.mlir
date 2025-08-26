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
  omp.parallel if(%n) if(%n) {
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

func.func @invalid_parent(%lb : index, %ub : index, %step : index) {
  // expected-error@+1 {{op expects parent op to be a loop wrapper}}
  omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
    omp.yield
  }
}

// -----

func.func @type_mismatch(%lb : index, %ub : index, %step : index) {
  omp.wsloop {
    // expected-error@+1 {{range argument type does not match corresponding IV type}}
    "omp.loop_nest" (%lb, %ub, %step) ({
    ^bb0(%iv2: i32):
      omp.yield
    }) : (index, index, index) -> ()
  }
}

// -----

func.func @iv_number_mismatch(%lb : index, %ub : index, %step : index) {
  omp.wsloop {
    // expected-error@+1 {{number of range arguments and IVs do not match}}
    "omp.loop_nest" (%lb, %ub, %step) ({
    ^bb0(%iv1 : index, %iv2 : index):
      omp.yield
    }) : (index, index, index) -> ()
  }
}

// -----

func.func @no_wrapper(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{op loop wrapper does not contain exactly one nested op}}
  omp.wsloop {
    %0 = arith.constant 0 : i32
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}

// -----

func.func @invalid_nested_wrapper(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{only supported nested wrapper is 'omp.simd'}}
  omp.wsloop {
    omp.distribute {
      omp.simd {
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          omp.yield
        }
      } {omp.composite}
    } {omp.composite}
  } {omp.composite}
}

// -----

func.func @no_loops(%lb : index, %ub : index, %step : index) {
  omp.wsloop {
    // expected-error@+1 {{op must represent at least one loop}}
    "omp.loop_nest" () ({
    ^bb0():
      omp.yield
    }) : () -> ()
  }
}

// -----

func.func @inclusive_not_a_clause(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{expected '{'}}
  omp.wsloop nowait inclusive {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}

// -----

func.func @order_value(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.wsloop order(default) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}

// -----
func.func @reproducible_order(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.wsloop order(reproducible:default) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}
// -----
func.func @unconstrained_order(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.wsloop order(unconstrained:default) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}
// -----

func.func @if_not_allowed(%lb : index, %ub : index, %step : index, %bool_var : i1) {
  // expected-error @below {{expected '{'}}
  omp.wsloop if(%bool_var) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}

// -----

func.func @num_threads_not_allowed(%lb : index, %ub : index, %step : index, %int_var : i32) {
  // expected-error @below {{expected '{'}}
  omp.wsloop num_threads(%int_var: i32) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}

// -----

func.func @proc_bind_not_allowed(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{expected '{'}}
  omp.wsloop proc_bind(close) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
}

// -----

llvm.func @test_omp_wsloop_dynamic_bad_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{unknown modifier type: ginandtonic}}
  omp.wsloop schedule(dynamic, ginandtonic) {
    omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_many_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{unexpected modifier(s)}}
  omp.wsloop schedule(dynamic, monotonic, monotonic, monotonic) {
    omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{incorrect modifier order}}
  omp.wsloop schedule(dynamic, simd, monotonic) {
    omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier2(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{incorrect modifier order}}
  omp.wsloop schedule(dynamic, monotonic, monotonic) {
    omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier3(%lb : i64, %ub : i64, %step : i64) -> () {
  // expected-error @+1 {{incorrect modifier order}}
  omp.wsloop schedule(dynamic, simd, simd) {
    omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

func.func @omp_simd() -> () {
  // expected-error @below {{op loop wrapper does not contain exactly one nested op}}
  omp.simd {}
  return
}

// -----

func.func @omp_simd_nested_wrapper(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{op must wrap an 'omp.loop_nest' directly}}
  omp.simd {
    omp.distribute {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
  }
  return
}

// -----

func.func @omp_simd_pretty_aligned(%lb : index, %ub : index, %step : index,
                                   %data_var : memref<i32>) -> () {
  //  expected-error @below {{expected '->'}}
  omp.simd aligned(%data_var : memref<i32>) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_simd_aligned_mismatch(%arg0 : index, %arg1 : index,
                                     %arg2 : index, %arg3 : memref<i32>,
                                     %arg4 : memref<i32>) -> () {
  //  expected-error @below {{op expected as many alignment values as aligned variables}}
  "omp.simd"(%arg3, %arg4) ({
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }) {alignments = [128],
      operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>} : (memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simd_aligned_negative(%arg0 : index, %arg1 : index,
                                     %arg2 : index, %arg3 : memref<i32>,
                                     %arg4 : memref<i32>) -> () {
  //  expected-error @below {{op alignment should be greater than 0}}
  "omp.simd"(%arg3, %arg4) ({
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }) {alignments = [-1, 128], operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>} : (memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simd_unexpected_alignment(%arg0 : index, %arg1 : index,
                                         %arg2 : index, %arg3 : memref<i32>,
                                         %arg4 : memref<i32>) -> () {
  //  expected-error @below {{unexpected alignment values attribute}}
  "omp.simd"() ({
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }) {alignments = [1, 128]} : () -> ()
  return
}

// -----

func.func @omp_simd_aligned_float(%arg0 : index, %arg1 : index,
                                  %arg2 : index, %arg3 : memref<i32>,
                                  %arg4 : memref<i32>) -> () {
  //  expected-error @below {{failed to satisfy constraint: 64-bit integer array attribute}}
  "omp.simd"(%arg3, %arg4) ({
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }) {alignments = [1.5, 128], operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>} : (memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simd_aligned_the_same_var(%arg0 : index, %arg1 : index,
                                         %arg2 : index, %arg3 : memref<i32>,
                                         %arg4 : memref<i32>) -> () {
  //  expected-error @below {{aligned variable used more than once}}
  "omp.simd"(%arg3, %arg3) ({
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }) {alignments = [1, 128], operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>} : (memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simd_nontemporal_the_same_var(%arg0 : index,  %arg1 : index,
                                             %arg2 : index,
                                             %arg3 : memref<i32>) -> () {
  //  expected-error @below {{nontemporal variable used more than once}}
  "omp.simd"(%arg3, %arg3) ({
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }) {operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 0, 0>} : (memref<i32>, memref<i32>) -> ()
  return
}

// -----

func.func @omp_simd_order_value(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.simd order(default) {
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_simd_reproducible_order(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.simd order(reproducible:default) {
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }
  return
}
// -----
func.func @omp_simd_unconstrained_order(%lb : index, %ub : index, %step : index) {
  // expected-error @below {{invalid clause value: 'default'}}
  omp.simd order(unconstrained:default) {
    omp.loop_nest (%iv) : index = (%arg0) to (%arg1) step (%arg2) {
      omp.yield
    }
  }
  return
}
// -----
func.func @omp_simd_pretty_simdlen(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{op attribute 'simdlen' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  omp.simd simdlen(0) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_simd_pretty_safelen(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{op attribute 'safelen' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  omp.simd safelen(0) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_simd_pretty_simdlen_safelen(%lb : index, %ub : index, %step : index) -> () {
  // expected-error @below {{op simdlen clause and safelen clause are both present, but the simdlen value is not less than or equal to safelen value}}
  omp.simd simdlen(2) safelen(1) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_simd_bad_privatizer(%lb : index, %ub : index, %step : index) {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  // expected-error @below {{Cannot find privatizer '@not_defined'}}
  omp.simd private(@not_defined %1 -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%arg2) : index = (%lb) to (%ub) inclusive step (%step) {
      omp.yield
    }
  }
}

// -----

omp.private {type = firstprivate} @_QFEp_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}
func.func @omp_simd_firstprivate(%lb : index, %ub : index, %step : index) {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  // expected-error @below {{FIRSTPRIVATE cannot be used with SIMD}}
  omp.simd private(@_QFEp_firstprivate_i32 %1 -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%arg2) : index = (%lb) to (%ub) inclusive step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// -----

// expected-error @below {{op expects alloc region to yield a value of the reduction type}}
omp.declare_reduction @add_f32 : f32
alloc {
^bb0(%arg: f32):
// nonsense test code
  %0 = arith.constant 0.0 : f64
  omp.yield (%0 : f64)
}
init {
^bb0(%arg0: f32, %arg1: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// -----

// expected-error @below {{op expects two arguments to the initializer region when an allocation region is used}}
omp.declare_reduction @add_f32 : f32
alloc {
^bb0(%arg: f32):
// nonsense test code
  omp.yield (%arg : f32)
}
init {
^bb0(%arg0: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// -----

// expected-error @below {{op expects one argument to the initializer region when no allocation region is used}}
omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32, %arg2: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// -----

// expected-error @below {{op expects initializer region argument to match the reduction type}}
omp.declare_reduction @add_f32 : f64
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
omp.declare_reduction @add_f32 : f32
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
omp.declare_reduction @add_f32 : f32
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
omp.declare_reduction @add_f32 : f32
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
omp.declare_reduction @add_f32 : f32
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
omp.declare_reduction @add_f32 : f32
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

// expected-error @below {{op expects cleanup region with one argument of the reduction type}}
omp.declare_reduction @add_f32 : f32
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
cleanup {
^bb0(%arg: f64):
  omp.yield
}

// -----

// expected-error @below {{op region #0 ('allocRegion') failed to verify constraint: region with at most 1 blocks}}
omp.declare_reduction @alloc_reduction : !llvm.ptr
alloc {
^bb0(%arg: !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  cf.br ^bb1(%0: !llvm.ptr)
^bb1(%ret: !llvm.ptr):
  omp.yield (%ret : !llvm.ptr)
}
init {
^bb0(%arg: !llvm.ptr):
  %cst = arith.constant 1.0 : f32
  llvm.store %cst, %arg : f32, !llvm.ptr
  omp.yield (%arg : !llvm.ptr)
}
combiner {
^bb1(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  %1 = llvm.load %arg1 : !llvm.ptr -> f32
  %2 = arith.addf %0, %1 : f32
  llvm.store %2, %arg0 : f32, !llvm.ptr
  omp.yield (%arg0 : !llvm.ptr)
}

// -----

func.func @foo(%lb : index, %ub : index, %step : index) {
  %c1 = arith.constant 1 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr

  // expected-error @below {{expected symbol reference @foo to point to a reduction declaration}}
  omp.wsloop reduction(@foo %0 -> %prv : !llvm.ptr) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %2 = arith.constant 2.0 : f32
      omp.yield
    }
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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
  omp.wsloop reduction(@add_f32 %0 -> %prv, @add_f32 %0 -> %prv1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %2 = arith.constant 2.0 : f32
      omp.yield
    }
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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
  omp.wsloop reduction(@add_f32 %mem -> %prv : memref<1xf32>) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      %2 = arith.constant 2.0 : f32
      omp.yield
    }
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

func.func @omp_ordered_region1(%x : i32) -> () {
  omp.distribute {
    omp.loop_nest (%i) : i32 = (%x) to (%x) step (%x) {
      // expected-error @below {{op must be nested inside of a worksharing, simd or worksharing simd loop}}
      omp.ordered.region {
        omp.terminator
      }
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_ordered_region2(%x : i32) -> () {
  omp.wsloop {
    omp.loop_nest (%i) : i32 = (%x) to (%x) step (%x) {
      // expected-error @below {{the enclosing worksharing-loop region must have an ordered clause}}
      omp.ordered.region {
        omp.terminator
      }
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_ordered_region3(%x : i32) -> () {
  omp.wsloop ordered(1) {
    omp.loop_nest (%i) : i32 = (%x) to (%x) step (%x) {
      // expected-error @below {{the enclosing loop's ordered clause must not have a parameter present}}
      omp.ordered.region {
        omp.terminator
      }
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_ordered1(%vec0 : i64) -> () {
  // expected-error @below {{op must be nested inside of a loop}}
  omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {doacross_num_loops = 1 : i64}
  return
}

// -----

func.func @omp_ordered2(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64) -> () {
  omp.distribute {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // expected-error @below {{op must be nested inside of a worksharing, simd or worksharing simd loop}}
      omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {doacross_num_loops = 1 : i64}
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_ordered3(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64) -> () {
  omp.wsloop {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // expected-error @below {{the enclosing worksharing-loop region must have an ordered clause}}
      omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {doacross_num_loops = 1 : i64}
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_ordered4(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64) -> () {
  omp.wsloop ordered(0) {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // expected-error @below {{the enclosing loop's ordered clause must have a parameter present}}
      omp.ordered depend_type(dependsink) depend_vec(%vec0 : i64) {doacross_num_loops = 1 : i64}
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_ordered5(%arg1 : i32, %arg2 : i32, %arg3 : i32, %vec0 : i64, %vec1 : i64) -> () {
  omp.wsloop ordered(1) {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // expected-error @below {{number of variables in depend clause does not match number of iteration variables in the doacross loop}}
      omp.ordered depend_type(dependsource) depend_vec(%vec0, %vec1 : i64, i64) {doacross_num_loops = 2 : i64}
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_atomic_read1(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{the hints omp_sync_hint_nonspeculative and omp_sync_hint_speculative cannot be combined.}}
  omp.atomic.read %v = %x hint(speculative, nonspeculative) : memref<i32>, memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read2(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{invalid clause value: 'xyz'}}
  omp.atomic.read %v = %x memory_order(xyz) : memref<i32>, memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read3(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{memory-order must not be acq_rel or release for atomic reads}}
  omp.atomic.read %v = %x memory_order(acq_rel) : memref<i32>, memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read4(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{memory-order must not be acq_rel or release for atomic reads}}
  omp.atomic.read %v = %x memory_order(release) : memref<i32>, memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read5(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{`memory_order` clause can appear at most once in the expansion of the oilist directive}}
  omp.atomic.read %v = %x memory_order(acquire) memory_order(relaxed) : memref<i32>, memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read6(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{`hint` clause can appear at most once in the expansion of the oilist directive}}
  omp.atomic.read %v =  %x hint(speculative) hint(contended) : memref<i32>, memref<i32>, i32
  return
}

// -----

func.func @omp_atomic_read6(%x: memref<i32>, %v: memref<i32>) {
  // expected-error @below {{read and write must not be to the same location for atomic reads}}
  omp.atomic.read %x =  %x hint(speculative) : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    omp.terminator
  }
  return
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{invalid sequence of operations in the capture region}}
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %y : memref<i32>, memref<i32>, i32
    omp.terminator
  }
}

// -----

func.func @omp_atomic_capture(%x: memref<i32>, %y: memref<i32>, %v: memref<i32>, %expr: i32) {
  omp.atomic.capture {
    // expected-error @below {{captured variable in atomic.read must be updated in second operation}}
    omp.atomic.read %v = %y : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x : memref<i32>, memref<i32>, i32
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
    omp.atomic.read %v = %x memory_order(seq_cst) : memref<i32>, memref<i32>, i32
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
    }) {operandSegmentSizes = array<i32: 1,0,0,0,0,0,0,0>} : (memref<i32>) -> ()
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
    }) {operandSegmentSizes = array<i32: 0,0,0,1,0,0,0,0>} : (i32) -> ()
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
  }) {operandSegmentSizes = array<i32: 1,0,0,0>} : (memref<i32>) -> ()
  return
}

// -----

func.func @omp_sections(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.sections" (%data_var) ({
  ^bb0(%arg0: memref<i32>):
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,0,1>} : (memref<i32>) -> ()
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

omp.declare_reduction @add_f32 : f32
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

func.func @omp_sections(%x : !llvm.ptr) {
  omp.sections reduction(@add_f32 %x -> %arg0 : !llvm.ptr) {
    // expected-error @below {{op expected at least 1 entry block argument(s)}}
    omp.section {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_single(%data_var : memref<i32>) -> () {
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.single" (%data_var) ({
    omp.barrier
  }) {operandSegmentSizes = array<i32: 1,0,0,0>} : (memref<i32>) -> ()
  return
}

// -----

func.func @omp_single_copyprivate(%data_var : memref<i32>) -> () {
  // expected-error @below {{inconsistent number of copyprivate vars (= 1) and functions (= 0), both must be equal}}
  "omp.single" (%data_var) ({
    omp.barrier
  }) {operandSegmentSizes = array<i32: 0,0,1,0>} : (memref<i32>) -> ()
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
  // expected-error @below {{'omp.task' op operand count (1) does not match with the total size (0) specified in attribute 'operandSegmentSizes'}}
    "omp.task"(%data_var) ({
      "omp.terminator"() : () -> ()
    }) {depend_kinds = [], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0>} : (memref<i32>) -> ()
   "func.return"() : () -> ()
}

// -----

func.func @omp_task(%ptr: !llvm.ptr) {
  // expected-error @below {{op expected symbol reference @add_f32 to point to a reduction declaration}}
  omp.task in_reduction(@add_f32 %ptr -> %arg0 : !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
}

// -----

omp.declare_reduction @add_f32 : f32
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
  omp.task in_reduction(@add_f32 %ptr -> %arg0, @add_f32 %ptr -> %arg1 : !llvm.ptr, !llvm.ptr) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
}

// -----

omp.declare_reduction @add_i32 : i32
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
  omp.task in_reduction(@add_i32 %mem -> %arg0 : memref<1xf32>) {
    // CHECK: "test.foo"() : () -> ()
    "test.foo"() : () -> ()
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel() {
  // expected-error @below {{Orphaned cancel construct}}
  omp.cancel cancellation_construct_type(parallel)
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

func.func @omp_cancel_taskloop() {
  omp.sections {
    // expected-error @below {{cancel taskgroup must appear inside a task region}}
    omp.cancel cancellation_construct_type(taskgroup)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancel3(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop nowait {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // expected-error @below {{A worksharing construct that is canceled must not have a nowait clause}}
      omp.cancel cancellation_construct_type(loop)
      // CHECK: omp.yield
      omp.yield
    }
  }
  return
}

// -----

func.func @omp_cancel4(%arg1 : i32, %arg2 : i32, %arg3 : i32) -> () {
  omp.wsloop ordered(1) {
    omp.loop_nest (%0) : i32 = (%arg1) to (%arg2) step (%arg3) {
      // expected-error @below {{A worksharing construct that is canceled must not have an ordered clause}}
      omp.cancel cancellation_construct_type(loop)
      // CHECK: omp.yield
      omp.yield
    }
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
  // expected-error @below {{Orphaned cancellation point}}
  omp.cancellation_point cancellation_construct_type(parallel)
  return
}

// -----

func.func @omp_cancellationpoint() {
  omp.sections {
    // expected-error @below {{cancellation point parallel must appear inside a parallel region}}
    omp.cancellation_point cancellation_construct_type(parallel)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancellationpoint1() {
  omp.parallel {
    // expected-error @below {{cancellation point sections must appear inside a sections region}}
    omp.cancellation_point cancellation_construct_type(sections)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancellationpoint2() {
  omp.sections {
    // expected-error @below {{cancellation point loop must appear inside a worksharing-loop region}}
    omp.cancellation_point cancellation_construct_type(loop)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

func.func @omp_cancellationpoint_taskgroup() {
  omp.sections {
    // expected-error @below {{cancellation point taskgroup must appear inside a task region}}
    omp.cancellation_point cancellation_construct_type(taskgroup)
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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

func.func @scan_test_2(%lb: i32, %ub: i32, %step: i32) {
  %test1f32 = "test.f32"() : () -> (!llvm.ptr)
  omp.wsloop reduction(mod:inscan, @add_f32 %test1f32 -> %arg1 : !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
  // expected-error @below {{Exactly one of EXCLUSIVE or INCLUSIVE clause is expected}}
       omp.scan
        omp.yield
    }
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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

func.func @scan_test_2(%lb: i32, %ub: i32, %step: i32) {
  %test1f32 = "test.f32"() : () -> (!llvm.ptr)
  omp.wsloop reduction(mod:inscan, @add_f32 %test1f32 -> %arg1 : !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
  // expected-error @below {{Exactly one of EXCLUSIVE or INCLUSIVE clause is expected}}
       omp.scan inclusive(%test1f32 : !llvm.ptr) exclusive(%test1f32: !llvm.ptr)
        omp.yield
    }
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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

func.func @scan_test_2(%lb: i32, %ub: i32, %step: i32) {
  %test1f32 = "test.f32"() : () -> (!llvm.ptr)
  omp.wsloop reduction(@add_f32 %test1f32 -> %arg1 : !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
  // expected-error @below {{SCAN directive needs to be enclosed within a parent worksharing loop construct or SIMD construct with INSCAN reduction modifier}}
       omp.scan inclusive(%test1f32 : !llvm.ptr)
        omp.yield
    }
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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

func.func @scan_test_2(%lb: i32, %ub: i32, %step: i32) {
  %test1f32 = "test.f32"() : () -> (!llvm.ptr)
  omp.taskloop reduction(mod:inscan, @add_f32 %test1f32 -> %arg1 : !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
  // expected-error @below {{SCAN directive needs to be enclosed within a parent worksharing loop construct or SIMD construct with INSCAN reduction modifier}}
       omp.scan inclusive(%test1f32 : !llvm.ptr)
        omp.yield
    }
  }
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testmemref = "test.memref"() : () -> (memref<i32>)
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.taskloop"(%testmemref) ({
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }) {operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>} : (memref<i32>) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%testf32, %testf32_2) ({
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }) {operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 2>, reduction_syms = [@add_f32]} : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%testf32) ({
  ^bb0(%arg0: !llvm.ptr):
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }) {operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>, reduction_syms = [@add_f32, @add_f32]} : (!llvm.ptr) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  %testf32_2 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%testf32, %testf32_2) ({
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }) {in_reduction_syms = [@add_f32], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>} : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testf32 = "test.f32"() : () -> (!llvm.ptr)
  // expected-error @below {{expected as many reduction symbol references as reduction variables}}
  "omp.taskloop"(%testf32) ({
  ^bb0(%arg0: !llvm.ptr):
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }) {in_reduction_syms = [@add_f32, @add_f32], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>} : (!llvm.ptr) -> ()
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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
  omp.taskloop nogroup reduction(@add_f32 %testf32 -> %arg0, @add_f32 %testf32_2 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }
  return
}

// -----

omp.declare_reduction @add_f32 : f32
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
  omp.taskloop in_reduction(@add_f32 %testf32 -> %arg0 : !llvm.ptr) reduction(@add_f32 %testf32 -> %arg1 : !llvm.ptr) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testi64 = "test.i64"() : () -> (i64)
  // expected-error @below {{the grainsize clause and num_tasks clause are mutually exclusive and may not appear on the same taskloop directive}}
  omp.taskloop grainsize(%testi64: i64) num_tasks(%testi64: i64) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testi64 = "test.i64"() : () -> (i64)
  // expected-error @below {{invalid grainsize modifier : 'strict1'}}
  omp.taskloop grainsize(strict1, %testi64: i64) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }
  return
}
// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  %testi64 = "test.i64"() : () -> (i64)
  // expected-error @below {{invalid num_tasks modifier : 'default'}}
  omp.taskloop num_tasks(default, %testi64: i64) {
    omp.loop_nest (%i, %j) : i32 = (%lb, %ub) to (%ub, %lb) step (%step, %step) {
      omp.yield
    }
  }
  return
}
// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  // expected-error @below {{op nested in loop wrapper is not another loop wrapper or `omp.loop_nest`}}
  omp.taskloop {
    %0 = arith.constant 0 : i32
  }
  return
}

// -----

func.func @taskloop(%lb: i32, %ub: i32, %step: i32) {
  // expected-error @below {{only supported nested wrapper is 'omp.simd'}}
  omp.taskloop {
    omp.distribute {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
  } {omp.composite}
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
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(delete) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{to, from, tofrom and alloc map types are permitted}}
  omp.target map_entries(%mapv -> %arg0: memref<?xi32>) {
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_data(%map1: memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)  map_clauses(delete) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{to, from, tofrom and alloc map types are permitted}}
  omp.target_data map_entries(%mapv : memref<?xi32>){}
  return
}

// -----

func.func @omp_target_data() {
  // expected-error @below {{At least one of map, use_device_ptr_vars, or use_device_addr_vars operand must be present}}
  omp.target_data {}
  return
}

// -----

func.func @omp_target_enter_data(%map1: memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(from) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{to and alloc map types are permitted}}
  omp.target_enter_data map_entries(%mapv : memref<?xi32>){}
  return
}

// -----

func.func @omp_target_enter_data_depend(%a: memref<?xi32>) {
  %0 = omp.map.info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32>
  // expected-error @below {{op expected as many depend values as depend variables}}
  omp.target_enter_data map_entries(%0: memref<?xi32> ) {operandSegmentSizes = array<i32: 1, 0, 0, 0>}
  return
}

// -----

func.func @omp_target_exit_data(%map1: memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>)   map_clauses(to) capture(ByRef) -> memref<?xi32> {name = ""}
  // expected-error @below {{from, release and delete map types are permitted}}
  omp.target_exit_data map_entries(%mapv : memref<?xi32>){}
  return
}

// -----

func.func @omp_target_exit_data_depend(%a: memref<?xi32>) {
  %0 = omp.map.info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32>
  // expected-error @below {{op expected as many depend values as depend variables}}
  omp.target_exit_data map_entries(%0: memref<?xi32> ) {operandSegmentSizes = array<i32: 1, 0, 0, 0>}
  return
}

// -----

func.func @omp_target_update_invalid_motion_type(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{at least one of to or from map types must be specified, other map types are not permitted}}
  omp.target_update map_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_type_2(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(delete) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{at least one of to or from map types must be specified, other map types are not permitted}}
  omp.target_update map_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(always, to) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{present, mapper and iterator map type modifiers are permitted}}
  omp.target_update map_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_2(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(close, to) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{present, mapper and iterator map type modifiers are permitted}}
  omp.target_update map_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_3(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(implicit, to) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{present, mapper and iterator map type modifiers are permitted}}
  omp.target_update map_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_4(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(implicit, tofrom) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{either to or from map types can be specified, not both}}
  omp.target_update map_entries(%mapv : memref<?xi32>)
  return
}

// -----

func.func @omp_target_update_invalid_motion_modifier_5(%map1 : memref<?xi32>) {
  %mapv = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32> {name = ""}
  %mapv2 = omp.map.info var_ptr(%map1 : memref<?xi32>, tensor<?xi32>) map_clauses(from) capture(ByRef) -> memref<?xi32> {name = ""}

  // expected-error @below {{either to or from map types can be specified, not both}}
  omp.target_update map_entries(%mapv, %mapv2 : memref<?xi32>, memref<?xi32>)
  return
}
llvm.mlir.global internal @_QFsubEx() : i32

// -----

func.func @omp_target_update_data_depend(%a: memref<?xi32>) {
  %0 = omp.map.info var_ptr(%a: memref<?xi32>, tensor<?xi32>) map_clauses(to) capture(ByRef) -> memref<?xi32>
  // expected-error @below {{op expected as many depend values as depend variables}}
  omp.target_update map_entries(%0: memref<?xi32> ) {operandSegmentSizes = array<i32: 1, 0, 0, 0>}
  return
}

// -----

func.func @omp_target_multiple_teams() {
  // expected-error @below {{target containing multiple 'omp.teams' nested ops}}
  omp.target {
    omp.teams {
      omp.terminator
    }
    omp.teams {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_host_eval(%x : !llvm.ptr) {
  // expected-error @below {{op host_eval argument illegal use in 'llvm.load' operation}}
  omp.target host_eval(%x -> %arg0 : !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_host_eval_teams(%x : i1) {
  // expected-error @below {{op host_eval argument only legal as 'num_teams' and 'thread_limit' in 'omp.teams'}}
  omp.target host_eval(%x -> %arg0 : i1) {
    omp.teams if(%arg0) {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_host_eval_parallel(%x : i32) {
  // expected-error @below {{op host_eval argument only legal as 'num_threads' in 'omp.parallel' when representing target SPMD}}
  omp.target host_eval(%x -> %arg0 : i32) {
    omp.parallel num_threads(%arg0 : i32) {
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_host_eval_loop1(%x : i32) {
  // expected-error @below {{op host_eval argument only legal as loop bounds and steps in 'omp.loop_nest' when trip count must be evaluated in the host}}
  omp.target host_eval(%x -> %arg0 : i32) {
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_host_eval_loop2(%x : i32) {
  // expected-error @below {{op host_eval argument only legal as loop bounds and steps in 'omp.loop_nest' when trip count must be evaluated in the host}}
  omp.target host_eval(%x -> %arg0 : i32) {
    omp.teams {
    ^bb0:
      %0 = arith.constant 0 : i1
      llvm.cond_br %0, ^bb1, ^bb2
    ^bb1:
      omp.distribute {
        omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
          omp.yield
        }
      }
      llvm.br ^bb2
    ^bb2:
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @omp_target_depend(%data_var: memref<i32>) {
  // expected-error @below {{op expected as many depend values as depend variables}}
    "omp.target"(%data_var) ({
      "omp.terminator"() : () -> ()
    }) {depend_kinds = [], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>} : (memref<i32>) -> ()
   "func.return"() : () -> ()
}

// -----

func.func @omp_distribute_schedule(%chunk_size : i32, %lb : i32, %ub : i32, %step : i32) -> () {
  // expected-error @below {{op chunk size set without dist_schedule_static being present}}
  "omp.distribute"(%chunk_size) <{operandSegmentSizes = array<i32: 0, 0, 1, 0>}> ({
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      "omp.yield"() : () -> ()
    }
  }) : (i32) -> ()
}

// -----

func.func @omp_distribute_allocate(%data_var : memref<i32>, %lb : i32, %ub : i32, %step : i32) -> () {
  // expected-error @below {{expected equal sizes for allocate and allocator variables}}
  "omp.distribute"(%data_var) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>}> ({
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      "omp.yield"() : () -> ()
    }
  }) : (memref<i32>) -> ()
}

// -----

func.func @omp_distribute_nested_wrapper(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{an 'omp.wsloop' nested wrapper is only allowed when a composite 'omp.parallel' is the direct parent}}
  omp.distribute {
    "omp.wsloop"() ({
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        "omp.yield"() : () -> ()
      }
    }) {omp.composite} : () -> ()
  } {omp.composite}
}

// -----

func.func @omp_distribute_nested_wrapper2(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{only supported nested wrappers are 'omp.simd' and 'omp.wsloop'}}
  omp.distribute {
    "omp.taskloop"() ({
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        "omp.yield"() : () -> ()
      }
    }) : () -> ()
  } {omp.composite}
}

// -----

func.func @omp_distribute_nested_wrapper3(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute missing from composite wrapper}}
  omp.distribute {
    "omp.simd"() ({
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        "omp.yield"() : () -> ()
      }
    }) {omp.composite} : () -> ()
  }
}

// -----

func.func @omp_distribute_nested_wrapper4(%lb: index, %ub: index, %step: index) -> () {
  omp.parallel {
    // expected-error @below {{an 'omp.wsloop' nested wrapper is only allowed when a composite 'omp.parallel' is the direct parent}}
    omp.distribute {
      "omp.wsloop"() ({
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          "omp.yield"() : () -> ()
        }
      }) {omp.composite} : () -> ()
    } {omp.composite}
    omp.terminator
  }
}

// -----

func.func @omp_distribute_order() -> () {
// expected-error @below {{invalid clause value: 'default'}}
  omp.distribute order(default) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
  }
  return
}
// -----
func.func @omp_distribute_reproducible_order() -> () {
// expected-error @below {{invalid clause value: 'default'}}
  omp.distribute order(reproducible:default) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
  }
  return
}
// -----
func.func @omp_distribute_unconstrained_order() -> () {
// expected-error @below {{invalid clause value: 'default'}}
  omp.distribute order(unconstrained:default) {
    omp.loop_nest (%iv) : i32 = (%arg0) to (%arg0) step (%arg0) {
      omp.yield
    }
  }
  return
}
// -----
omp.private {type = private} @x.privatizer : i32 init {
^bb0(%arg0: i32, %arg1: i32):
  %0 = arith.constant 0.0 : f32
  // expected-error @below {{Invalid yielded value. Expected type: 'i32', got: 'f32'}}
  omp.yield(%0 : f32)
}

// -----

// expected-error @below {{Region argument type mismatch: got 'f32' expected 'i32'.}}
omp.private {type = private} @x.privatizer : i32 init {
^bb0(%arg0: i32, %arg1: f32):
  omp.yield
}

// -----

omp.private {type = private} @x.privatizer : f32 init {
^bb0(%arg0: f32, %arg1: f32):
  omp.yield(%arg0: f32)
} dealloc {
^bb0(%arg0: f32):
  // expected-error @below {{Did not expect any values to be yielded.}}
  omp.yield(%arg0 : f32)
}

// -----

omp.private {type = private} @x.privatizer : i32 init {
^bb0(%arg0: i32, %arg1: i32):
  // expected-error @below {{expected exit block terminator to be an `omp.yield` op.}}
  omp.terminator
}

// -----

// expected-error @below {{`init`: expected 2 region arguments, got: 1}}
omp.private {type = private} @x.privatizer : f32 init {
^bb0(%arg0: f32):
  omp.yield(%arg0 : f32)
}

// -----

// expected-error @below {{`copy`: expected 2 region arguments, got: 1}}
omp.private {type = firstprivate} @x.privatizer : f32 copy {
^bb0(%arg0: f32):
  omp.yield(%arg0 : f32)
}

// -----

// expected-error @below {{`dealloc`: expected 1 region arguments, got: 2}}
omp.private {type = private} @x.privatizer : f32 dealloc {
^bb0(%arg0: f32, %arg1: f32):
  omp.yield
}

// -----

// expected-error @below {{`private` clauses do not require a `copy` region.}}
omp.private {type = private} @x.privatizer : f32 copy {
^bb0(%arg0: f32, %arg1 : f32):
  omp.yield(%arg0 : f32)
}

// -----

// expected-error @below {{`firstprivate` clauses require at least a `copy` region.}}
omp.private {type = firstprivate} @x.privatizer : f32 init {
^bb0(%arg0: f32, %arg1: f32):
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

omp.private {type = private} @var1.privatizer : index init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
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

omp.private {type = firstprivate} @var1.privatizer : index copy {
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
  "omp.parallel"(%arg0) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 1, 0>, private_syms = [@x.privatizer, @y.privatizer]}> ({
    ^bb0(%arg2: !llvm.ptr):
      omp.terminator
    }) : (!llvm.ptr) -> ()
  return
}

// -----

omp.private {type = private} @var1.privatizer : !llvm.ptr copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

func.func @byref_in_private(%arg0: index) {
  // expected-error @below {{expected attribute value}}
  // expected-error @below {{custom op 'omp.parallel' invalid `private` format}}
  omp.parallel private(byref @var1.privatizer %arg0 -> %arg2 : index) {
    omp.terminator
  }

  return
}

// -----
func.func @masked_arg_type_mismatch(%arg0: f32) {
  // expected-error @below {{'omp.masked' op operand #0 must be integer or index, but got 'f32'}}
  "omp.masked"(%arg0) ({
      omp.terminator
    }) : (f32) -> ()
  return
}

// -----
func.func @masked_arg_count_mismatch(%arg0: i32, %arg1: i32) {
  // expected-error @below {{'omp.masked' op operand group starting at #0 requires 0 or 1 element, but found 2}}
  "omp.masked"(%arg0, %arg1) ({
      omp.terminator
    }) : (i32, i32) -> ()
  return
}

// -----
func.func @omp_parallel_missing_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute missing from composite operation}}
  omp.parallel {
    omp.distribute {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// -----
func.func @omp_parallel_invalid_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute present in non-composite operation}}
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
    omp.terminator
  } {omp.composite}
  return
}

// -----
func.func @omp_parallel_invalid_composite2(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{unexpected OpenMP operation inside of composite 'omp.parallel': omp.barrier}}
  omp.parallel {
    omp.barrier
    omp.distribute {
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          omp.yield
        }
      } {omp.composite}
    } {omp.composite}
    omp.terminator
  } {omp.composite}
  return
}

// -----
func.func @omp_parallel_invalid_composite3(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{multiple 'omp.distribute' nested inside of 'omp.parallel'}}
  omp.parallel {
    omp.distribute {
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          omp.yield
        }
      } {omp.composite}
    } {omp.composite}
    omp.distribute {
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          omp.yield
        }
      } {omp.composite}
    } {omp.composite}
    omp.terminator
  } {omp.composite}
  return
}

// -----
func.func @omp_wsloop_missing_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute missing from composite wrapper}}
  omp.wsloop {
    omp.simd {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    } {omp.composite}
  }
  return
}

// -----
func.func @omp_wsloop_invalid_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute present in non-composite wrapper}}
  omp.wsloop {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  } {omp.composite}
  return
}

// -----
func.func @omp_wsloop_missing_composite_2(%lb: index, %ub: index, %step: index) -> () {
  omp.parallel {
    omp.distribute {
      // expected-error @below {{'omp.composite' attribute missing from composite wrapper}}
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          omp.yield
        }
      }
    } {omp.composite}
    omp.terminator
  } {omp.composite}
  return
}

// -----
func.func @omp_simd_missing_composite(%lb: index, %ub: index, %step: index) -> () {
  omp.wsloop {
    // expected-error @below {{'omp.composite' attribute missing from composite wrapper}}
    omp.simd {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    }
  } {omp.composite}
  return
}

// -----
func.func @omp_simd_invalid_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute present in non-composite wrapper}}
  omp.simd {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  } {omp.composite}
  return
}

// -----
func.func @omp_distribute_missing_composite(%lb: index, %ub: index, %step: index) -> () {
  omp.parallel {
    // expected-error @below {{'omp.composite' attribute missing from composite wrapper}}
    omp.distribute {
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
          omp.yield
        }
      } {omp.composite}
    }
    omp.terminator
  } {omp.composite}
  return
}

// -----
func.func @omp_distribute_invalid_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute present in non-composite wrapper}}
  omp.distribute {
    omp.loop_nest (%0) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  } {omp.composite}
  return
}

// -----
func.func @omp_taskloop_missing_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute missing from composite wrapper}}
  omp.taskloop {
    omp.simd {
      omp.loop_nest (%i) : index = (%lb) to (%ub) step (%step)  {
        omp.yield
      }
    } {omp.composite}
  }
  return
}

// -----
func.func @omp_taskloop_invalid_composite(%lb: index, %ub: index, %step: index) -> () {
  // expected-error @below {{'omp.composite' attribute present in non-composite wrapper}}
  omp.taskloop {
    omp.loop_nest (%i) : index = (%lb) to (%ub) step (%step)  {
      omp.yield
    }
  } {omp.composite}
  return
}

// -----

func.func @omp_loop_invalid_nesting(%lb : index, %ub : index, %step : index) {

  // expected-error @below {{'omp.loop' op expected to be a standalone loop wrapper}}
  omp.loop {
    omp.simd {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    } {omp.composite}
  }

  return
}

// -----

func.func @omp_loop_invalid_nesting2(%lb : index, %ub : index, %step : index) {

  omp.simd {
    // expected-error @below {{'omp.loop' op expected to be a standalone loop wrapper}}
    omp.loop {
      omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
        omp.yield
      }
    } {omp.composite}
  }

  return
}

// -----

func.func @omp_loop_invalid_binding(%lb : index, %ub : index, %step : index) {

  // expected-error @below {{custom op 'omp.loop' invalid clause value: 'dummy_value'}}
  omp.loop bind(dummy_value) {
    omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  return
}

// -----
func.func @nested_wrapper(%idx : index) {
  omp.workshare {
    // expected-error @below {{'omp.workshare.loop_wrapper' op expected to be a standalone loop wrapper}}
    omp.workshare.loop_wrapper {
      omp.simd {
        omp.loop_nest (%iv) : index = (%idx) to (%idx) step (%idx) {
          omp.yield
        }
      } {omp.composite}
    }
    omp.terminator
  }
  return
}

// -----
func.func @not_wrapper() {
  omp.workshare {
    // expected-error @below {{op nested in loop wrapper is not another loop wrapper or `omp.loop_nest`}}
    omp.workshare.loop_wrapper {
      %0 = arith.constant 0 : index
    }
    omp.terminator
  }
  return
}

// -----
func.func @missing_workshare(%idx : index) {
  // expected-error @below {{must be nested in an omp.workshare}}
  omp.workshare.loop_wrapper {
    omp.loop_nest (%iv) : index = (%idx) to (%idx) step (%idx) {
      omp.yield
    }
  }
  return
}

// -----
  // expected-error @below {{op expected terminator to be a DeclareMapperInfoOp}}
  omp.declare_mapper @missing_declareMapperInfo : !llvm.struct<"mytype", (array<1024 x i32>)> {
  ^bb0(%arg0: !llvm.ptr):
    omp.terminator
  }

// -----
llvm.func @invalid_mapper(%0 : !llvm.ptr) {
  // expected-error @below {{invalid mapper id}}
  %1 = omp.map.info var_ptr(%0 : !llvm.ptr, !llvm.struct<"my_type", (i32)>) map_clauses(to) capture(ByRef) mapper(@my_mapper) -> !llvm.ptr {name = ""}
  omp.target_data map_entries(%1 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----
func.func @invalid_allocate_align_1(%arg0 : memref<i32>) -> () {
  // expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  omp.allocate_dir (%arg0 : memref<i32>) align(-1)

  return
}

// -----
func.func @invalid_allocate_align_2(%arg0 : memref<i32>) -> () {
  // expected-error @below {{must be power of 2}}
  omp.allocate_dir (%arg0 : memref<i32>) align(3)

  return
}

// -----
func.func @invalid_allocate_allocator(%arg0 : memref<i32>) -> () {
  // expected-error @below {{invalid clause value}}
  omp.allocate_dir (%arg0 : memref<i32>) allocator(omp_small_cap_mem_alloc)

  return
}

// -----
func.func @invalid_workdistribute_empty_region() -> () {
  omp.teams {
    // expected-error @below {{region cannot be empty}}
    omp.workdistribute {
    }
    omp.terminator
  }
  return
}

// -----
func.func @invalid_workdistribute_no_terminator() -> () {
  omp.teams {
    // expected-error @below {{region must be terminated with omp.terminator}}
    omp.workdistribute {
      %c0 = arith.constant 0 : i32
    }
    omp.terminator
  }
  return
}

// -----
func.func @invalid_workdistribute_wrong_terminator() -> () {
  omp.teams {
    // expected-error @below {{region must be terminated with omp.terminator}}
    omp.workdistribute {
      %c0 = arith.constant 0 : i32
      func.return
    }
    omp.terminator
  }
  return
}

// -----
func.func @invalid_workdistribute_multiple_terminators() -> () {
  omp.teams {
    // expected-error @below {{region must have exactly one terminator}}
    omp.workdistribute {
      %cond = arith.constant true
      cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
      omp.terminator
    ^bb2:
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----
func.func @invalid_workdistribute_with_barrier() -> () {
  omp.teams {
    // expected-error @below {{explicit barriers are not allowed in workdistribute region}}
    omp.workdistribute {
      %c0 = arith.constant 0 : i32
      omp.barrier
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----
func.func @invalid_workdistribute_nested_parallel() -> () {
  omp.teams {
    // expected-error @below {{nested parallel constructs not allowed in workdistribute}}
    omp.workdistribute {
      omp.parallel {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----
// Test: nested teams not allowed in workdistribute
func.func @invalid_workdistribute_nested_teams() -> () {
  omp.teams {
    // expected-error @below {{nested teams constructs not allowed in workdistribute}}
    omp.workdistribute {
      omp.teams {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----
func.func @invalid_workdistribute() -> () {
// expected-error @below {{workdistribute must be nested under teams}}
  omp.workdistribute {
    omp.terminator
  }
  return
}
