// RUN: mlir-translate --export-smtlib %s --split-input-file --verify-diagnostics

smt.solver () : () -> () {
  %0 = smt.constant true
  // expected-error @below {{must not have any result values}}
  %1 = smt.check sat {
    smt.yield %0 : !smt.bool
  } unknown {
    smt.yield %0 : !smt.bool
  } unsat {
    smt.yield %0 : !smt.bool
  } -> !smt.bool
}

// -----

smt.solver () : () -> () {
  // expected-error @below {{'sat' region must be empty}}
  smt.check sat {
    %0 = smt.constant true
    smt.yield
  } unknown {
  } unsat {
  }
}

// -----

smt.solver () : () -> () {
  // expected-error @below {{'unknown' region must be empty}}
  smt.check sat {
  } unknown {
    %0 = smt.constant true
    smt.yield
  } unsat {
  }
}

// -----

smt.solver () : () -> () {
  // expected-error @below {{'unsat' region must be empty}}
  smt.check sat {
  } unknown {
  } unsat {
    %0 = smt.constant true
    smt.yield
  }
}

// -----

// expected-error @below {{solver scopes with inputs or results are not supported}}
%0 = smt.solver () : () -> (i1) {
  %1 = arith.constant true
  smt.yield %1 : i1
}

// -----

smt.solver () : () -> () {
  // expected-error @below {{solver must not contain any non-SMT operations}}
  %1 = arith.constant true
}

// -----

func.func @solver_input(%arg0: i1) {
  // expected-error @below {{solver scopes with inputs or results are not supported}}
  smt.solver (%arg0) : (i1) -> () {
  ^bb0(%arg1: i1):
    smt.yield
  }
  return
}

// -----

smt.solver () : () -> () {
  %0 = smt.declare_fun : !smt.sort<"uninterpreted0">
  // expected-error @below {{uninterpreted sorts with same identifier but different arity found}}
  %1 = smt.declare_fun : !smt.sort<"uninterpreted0"[!smt.bool]>
}
