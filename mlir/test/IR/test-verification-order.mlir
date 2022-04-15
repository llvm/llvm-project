// RUN: mlir-opt --mlir-disable-threading -split-input-file -verify-diagnostics %s

func @verify_operand_type() {
  %0 = arith.constant 1 : index
  // expected-error@+1 {{op operand #0 must be 32-bit signless integer, but got 'index'}}
  "test.verifiers"(%0) ({
    %1 = arith.constant 2 : index
  }) : (index) -> ()
  return
}

// -----

func @verify_nested_op_block_trait() {
  %0 = arith.constant 1 : i32
  // expected-remark@+1 {{success run of verifier}}
  "test.verifiers"(%0) ({
    %1 = arith.constant 2 : index
    // expected-error@+1 {{op requires one region}}
    "test.verifiers"(%1) : (index) -> ()
  }) : (i32) -> ()
  return
}

// -----

func @verify_nested_op_operand() {
  %0 = arith.constant 1 : i32
  // expected-remark@+1 {{success run of verifier}}
  "test.verifiers"(%0) ({
    %1 = arith.constant 2 : index
    // expected-error@+1 {{op operand #0 must be 32-bit signless integer, but got 'index'}}
    "test.verifiers"(%1) ({
      %2 = arith.constant 3 : index
    }) : (index) -> ()
  }) : (i32) -> ()
  return
}

// -----

func @verify_nested_isolated_above() {
  %0 = arith.constant 1 : i32
  // expected-remark@+1 {{success run of verifier}}
  "test.verifiers"(%0) ({
    // expected-remark@-1 {{success run of region verifier}}
    %1 = arith.constant 2 : i32
    // expected-remark@+1 {{success run of verifier}}
    "test.verifiers"(%1) ({
      // expected-remark@-1 {{success run of region verifier}}
      %2 = arith.constant 3 : index
    }) : (i32) -> ()
  }) : (i32) -> ()
  return
}
