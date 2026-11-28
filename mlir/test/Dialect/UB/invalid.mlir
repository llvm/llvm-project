// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @invalid_predicate(%value: i32) -> i32 {
  // expected-error @+1 {{Invalid attribute `predicate` in property conversion}}
  %assumed = "ub.assume"(%value) <{predicate = 0 : i32}> : (i32) -> i32
  return %assumed : i32
}

// -----

func.func @mismatched_result(%value: i32) -> i64 {
  // expected-error @+1 {{failed to verify that all of {value, result} have same type}}
  %assumed = "ub.assume"(%value) {predicate = #arith.assume_cmpi<sgt, 0 : i32>} : (i32) -> i64
  return %assumed : i64
}

// -----

func.func @mismatched_rhs(%value: i32) -> i32 {
  // expected-error @+1 {{arith.assume_cmpi expected predicated value type 'i64', but got 'i32'}}
  %assumed = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i64>() : i32
  return %assumed : i32
}

// -----

func.func @unexpected_operand(%value: i32, %rhs: i32) -> i32 {
  // expected-error @+1 {{arith.assume_cmpi does not accept predicate operands}}
  %assumed = ub.assume %value, #arith.assume_cmpi<slt, 5 : i32>(%rhs : i32) : i32
  return %assumed : i32
}

// -----

func.func @non_integer_predicated_value(%value: f32) -> f32 {
  // expected-error @+1 {{arith.assume_cmpi expected predicated value type 'i32', but got 'f32'}}
  %assumed = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : f32
  return %assumed : f32
}

// -----

// expected-error @+1 {{expected a non-zero-width signless integer RHS}}
module attributes {predicate = #arith.assume_cmpi<sgt, 0 : index>} {}

// -----

// expected-error @+1 {{expected a non-zero-width signless integer RHS}}
module attributes {predicate = #arith.assume_cmpi<sgt, 0 : si32>} {}
