// REQUIRES: z3-prover
// RUN: mlir-translate --export-smtlib %s | z3 -in 2>&1 | FileCheck %s
// RUN: mlir-translate --export-smtlib --smtlibexport-inline-single-use-values %s | z3 -in 2>&1 | FileCheck %s

// Quantifiers Attributes
smt.solver () : () -> () {

  %true = smt.constant true

  %7 = smt.exists weight 2 {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  }
  smt.assert %7

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Quantifiers Attributes
smt.solver () : () -> () {

  %true = smt.constant true

  %7 = smt.exists weight 2 {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %4: !smt.bool
  }
  smt.assert %7

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

// Quantifiers Attributes
smt.solver () : () -> () {

  %true = smt.constant true

  %7 = smt.exists {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %4: !smt.bool
  }
  smt.assert %7

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

smt.solver () : () -> () {

  %true = smt.constant true
  %three = smt.int.constant 3
  %four = smt.int.constant 4

  %7 = smt.exists {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %three: !smt.int
    %5 = smt.eq %arg3, %four: !smt.int
    %6 = smt.eq %4, %5: !smt.bool
    smt.yield %6 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %three: !smt.int
    smt.yield %4: !smt.bool}, {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %5 = smt.eq %arg3, %four: !smt.int
    smt.yield %5: !smt.bool
  }
  smt.assert %7

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat

smt.solver () : () -> () {

  %true = smt.constant true
  %three = smt.int.constant 3
  %four = smt.int.constant 4

  %10 = smt.exists {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %three: !smt.int
    %5 = smt.eq %arg3, %four: !smt.int
    %9 = smt.eq %4, %5: !smt.bool
    smt.yield %9 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %three: !smt.int
    %5 = smt.eq %arg3, %four: !smt.int
    smt.yield %4, %5: !smt.bool, !smt.bool
  }
  smt.assert %10

  smt.check sat {} unknown {} unsat {}

}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat
