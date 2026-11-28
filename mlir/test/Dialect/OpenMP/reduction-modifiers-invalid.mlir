// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

omp.declare_reduction @add : i32 init {
^bb0(%v: i32):
  %zero = arith.constant 0 : i32
  omp.yield(%zero : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %sum = arith.addi %a, %b : i32
  omp.yield(%sum : i32)
}

func.func @invalid(%x: !llvm.ptr, %y: !llvm.ptr) {
  // expected-error@+1 {{expected as many reduction modifiers as reduction variables}}
  "omp.parallel"(%x, %y) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 2>, reduction_syms = [@add, @add], reduction_mods = [#omp.reduction_modifier<task>]}> ({
  ^bb0(%px: !llvm.ptr, %py: !llvm.ptr):
    omp.terminator
  }) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

omp.declare_reduction @add : i32 init {
^bb0(%v: i32):
  %zero = arith.constant 0 : i32
  omp.yield(%zero : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %sum = arith.addi %a, %b : i32
  omp.yield(%sum : i32)
}

func.func @invalid(%x: !llvm.ptr, %y: !llvm.ptr) {
  // expected-error@+1 {{'reduction_mods' failed to satisfy constraint: array of reduction modifiers}}
  "omp.parallel"(%x, %y) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 2>, reduction_syms = [@add, @add], reduction_mods = [42 : i32, #omp.reduction_modifier<task>]}> ({
  ^bb0(%px: !llvm.ptr, %py: !llvm.ptr):
    omp.terminator
  }) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @simd_empty_vars() {
  // expected-error@+1 {{expected as many reduction modifiers as reduction variables}}
  "omp.simd"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, reduction_mods = [#omp.reduction_modifier<inscan>]}> ({
    omp.terminator
  }) : () -> ()
  return
}

// -----

func.func @removed_modifier_keyword(%x: !llvm.ptr) {
  // expected-error@+2 {{invalid `reduction` format}}
  // expected-error@+1 {{expected attribute value}}
  omp.parallel reduction(mod: task, @add %x -> %px : !llvm.ptr) {
    omp.terminator
  }
  return
}
