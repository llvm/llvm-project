// REQUIRES: z3-prover
// RUN: mlir-translate --export-smtlib %s | z3 -in 2>&1 | FileCheck %s
// RUN: mlir-translate --export-smtlib --smtlibexport-inline-single-use-values %s | z3 -in 2>&1 | FileCheck %s

// Function and constant symbol declarations, uninterpreted sorts
smt.solver () : () -> () {
  %0 = smt.declare_fun "b" : !smt.bv<32>
  %1 = smt.declare_fun : !smt.int
  %2 = smt.declare_fun : !smt.func<(!smt.int, !smt.bv<32>) !smt.bool>
  %3 = smt.apply_func %2(%1, %0) : !smt.func<(!smt.int, !smt.bv<32>) !smt.bool>
  smt.assert %3

  %4 = smt.declare_fun : !smt.sort<"uninterpreted">
  %5 = smt.declare_fun : !smt.sort<"other"[!smt.sort<"uninterpreted">]>
  %6 = smt.declare_fun : !smt.func<(!smt.sort<"other"[!smt.sort<"uninterpreted">]>, !smt.sort<"uninterpreted">) !smt.bool>
  %7 = smt.apply_func %6(%5, %4) : !smt.func<(!smt.sort<"other"[!smt.sort<"uninterpreted">]>, !smt.sort<"uninterpreted">) !smt.bool>
  smt.assert %7

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Big expression
smt.solver () : () -> () {
  %true = smt.constant true
  %false = smt.constant false
  %0 = smt.not %true
  %1 = smt.and %0, %true, %false
  %2 = smt.or %1, %0, %true
  %3 = smt.xor %0, %2
  %4 = smt.implies %3, %false
  %5 = smt.implies %4, %true
  smt.assert %5

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Constant array
smt.solver () : () -> () {
  %true = smt.constant true
  %false = smt.constant false
  %c = smt.int.constant 0
  %0 = smt.array.broadcast %true : !smt.array<[!smt.int -> !smt.bool]>
  %1 = smt.array.store %0[%c], %false : !smt.array<[!smt.int -> !smt.bool]>
  %2 = smt.array.select %1[%c] : !smt.array<[!smt.int -> !smt.bool]>
  smt.assert %2

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK: unsat

// BitVector extract and concat, and constants
smt.solver () : () -> () {
  %xf = smt.bv.constant #smt.bv<0xf> : !smt.bv<4>
  %x0 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
  %xff = smt.bv.constant #smt.bv<0xff> : !smt.bv<8>

  %0 = smt.bv.concat %xf, %x0 : !smt.bv<4>, !smt.bv<4>
  %1 = smt.bv.extract %0 from 4 : (!smt.bv<8>) -> !smt.bv<4>
  %2 = smt.bv.repeat 2 times %1 : !smt.bv<4>
  %3 = smt.eq %2, %xff : !smt.bv<8>
  smt.assert %3

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Quantifiers
smt.solver () : () -> () {
  %1 = smt.forall ["a"] {
  ^bb0(%arg2: !smt.int):
    %c2_1 = smt.int.constant 2
    %2 = smt.int.mul %arg2, %c2_1

    %3 = smt.exists {
    ^bb0(%arg1: !smt.int):
      %c2 = smt.int.constant 2
      %4 = smt.int.mul %c2, %arg1
      %5 = smt.eq %4, %2 : !smt.int
      smt.yield %5 : !smt.bool
    }

    smt.yield %3 : !smt.bool
  }
  smt.assert %1

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Push and pop
smt.solver () : () -> () {
  %three = smt.int.constant 3
  %four = smt.int.constant 4
  %unsat_eq = smt.eq %three, %four : !smt.int
  %sat_eq = smt.eq %three, %three : !smt.int

  smt.push 1
  smt.assert %unsat_eq
  smt.check sat {} unknown {} unsat {}
  smt.pop 1
  smt.assert %sat_eq
  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: sat
// CHECK: unsat
// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Reset
smt.solver () : () -> () {
  %three = smt.int.constant 3
  %four = smt.int.constant 4
  %unsat_eq = smt.eq %three, %four : !smt.int
  %sat_eq = smt.eq %three, %three : !smt.int

  smt.assert %unsat_eq
  smt.check sat {} unknown {} unsat {}
  smt.reset
  smt.assert %sat_eq
  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: sat
// CHECK: unsat
// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

smt.solver () : () -> () {
  smt.set_logic "HORN"
  %c = smt.declare_fun : !smt.int
  %c4 = smt.int.constant 4
  %eq = smt.eq %c, %c4 : !smt.int
  smt.assert %eq
  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK-NOT: sat
// CHECK: unknown
