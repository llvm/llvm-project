// RUN: mlir-opt -split-input-file -verify-diagnostics %s

omp.private {type = private} @x.privatizer : i32

func.func @blockarg(%x : i32) {
  // expected-error @below {{op expected at least 2 entry block argument(s)}}
  "omp.parallel" (%x, %x) ({
  ^bb0(%arg0 : i32):
    omp.terminator
  }) {operandSegmentSizes = array<i32: 0,0,0,0,2,0>,
      private_syms = [@x.privatizer, @x.privatizer]} : (i32, i32) -> ()
  return
}

// -----

func.func @loopwrapper_multiple(%x : i32) {
  // expected-error @below {{op loop wrapper does not contain exactly one nested op}}
  omp.wsloop {
    omp.simd {
      omp.loop_nest (%iv) : i32 = (%x) to (%x) step (%x) {
        omp.yield
      }
    } {omp.composite}
    llvm.call @foo() : () -> ()
  } {omp.composite}
  return
}

// -----

func.func @loopwrapper_invalid() {
  // expected-error @below {{op nested in loop wrapper is not another loop wrapper or `omp.loop_nest`}}
  omp.wsloop {
    omp.taskyield
  }
  return
}

// -----

func.func @composable_ineligible_composite() {
  // expected-error @below {{op non-loop wrapper cannot be composite}}
  omp.task {
    omp.terminator
  } {omp.composite}
  return
}

// -----

func.func @composable_multiple_combined() {
  // expected-error @below {{op multiple eligible child ops found in combined op}}
  omp.teams {
    omp.parallel {
      omp.terminator
    }
    omp.single {
      omp.terminator
    }
    omp.terminator
  } {omp.combined}
  return
}

// -----

func.func @composable_loop_combined(%x : i32) {
  // expected-error @below {{op nested combined child op is part of a loop}}
  omp.parallel {
  ^bb0:
    llvm.br ^bb1
  ^bb1:
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%x) to (%x) step (%x) {
        omp.yield
      }
    }
    %0 = arith.constant 0 : i1
    llvm.cond_br %0, ^bb1, ^bb2
  ^bb2:
    omp.terminator
  } {omp.combined}
  return
}

// -----

func.func @composable_conditional_combined(%x : i32) {
  // expected-error @below {{op nested combined child op doesn't unconditionally execute}}
  omp.parallel {
  ^bb0:
    %0 = arith.constant 0 : i1
    llvm.cond_br %0, ^bb1, ^bb2
  ^bb1:
    omp.wsloop {
      omp.loop_nest (%iv) : i32 = (%x) to (%x) step (%x) {
        omp.yield
      }
    }
    llvm.br ^bb2
  ^bb2:
    omp.terminator
  } {omp.combined}
  return
}

// -----

// expected-error @below {{omp.declare_target can only be applied to DeclareTargetInterface ops}}
%0 = arith.constant { omp.declare_target = #omp.declaretarget<capture_clause = enter> } 2 : i32

// -----

// expected-error @below {{omp.declare_target must be an #omp.declaretarget attribute}}
func.func private @declare_target_attr_type() attributes { omp.declare_target = 10 : i32 }

// -----

// expected-error @below {{omp.declare_target 'automap' is not valid on functions}}
func.func private @declare_target_automap() attributes { omp.declare_target = #omp.declaretarget<automap = true>}

// -----

// expected-error @below {{omp.declare_target 'link' is not valid on functions}}
func.func private @declare_target_link() attributes { omp.declare_target = #omp.declaretarget<capture_clause = link>}

// -----

// expected-error @below {{omp.declare_target 'implicit' is only valid on functions}}
llvm.mlir.global @declare_target_implicit() {omp.declare_target = #omp.declaretarget<implicit = true>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}
