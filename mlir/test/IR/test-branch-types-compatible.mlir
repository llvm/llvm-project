// RUN: mlir-opt %s --split-input-file --verify-diagnostics

// RegionBranchOpInterface: compatible integer types (i32 <-> i64) should pass.
func.func @region_branch_compat() -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = "test.region_types_compat"(%c0) ({
  ^bb0(%arg0: i64):
    %c1 = arith.constant 1 : i64
    "test.types_compat_yield"(%c1) : (i64) -> ()
  }) : (i32) -> i32
  return %0 : i32
}

// -----

// RegionBranchOpInterface: incompatible types (i32 <-> f32) should fail.
func.func @region_branch_incompat() -> i32 {
  %c0 = arith.constant 0 : i32
  // expected-error @+2 {{along control flow edge from parent to Region #0: successor operand type #0 'i32' should match successor input type #0 'f32'}}
  // expected-note @+1 {{region branch point}}
  %0 = "test.region_types_compat"(%c0) ({
  ^bb0(%arg0: f32):
    %c1 = arith.constant 1.0 : f32
    "test.types_compat_yield"(%c1) : (f32) -> ()
  }) : (i32) -> i32
  return %0 : i32
}

// -----

// LoopLikeOpInterface: compatible integer types (i32 <-> i64) should pass.
func.func @loop_compat() -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = "test.loop_types_compat"(%c0) ({
  ^bb0(%arg0: i64):
    %c1 = arith.constant 1 : i64
    "test.types_compat_yield"(%c1) : (i64) -> ()
  }) : (i32) -> i32
  return %0 : i32
}

// -----

// LoopLikeOpInterface: incompatible init vs iter_arg (i32 <-> f32) should fail.
func.func @loop_incompat_init() -> i32 {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{0-th init and 0-th region iter_arg have different type: 'i32' != 'f32'}}
  %0 = "test.loop_types_compat"(%c0) ({
  ^bb0(%arg0: f32):
    %c1 = arith.constant 1.0 : f32
    "test.types_compat_yield"(%c1) : (f32) -> ()
  }) : (i32) -> i32
  return %0 : i32
}

// -----

// LoopLikeOpInterface: incompatible iter_arg vs yield (i32 <-> f32) should fail.
func.func @loop_incompat_yield() -> i32 {
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{0-th region iter_arg and 0-th yielded value have different type: 'i32' != 'f32'}}
  %0 = "test.loop_types_compat"(%c0) ({
  ^bb0(%arg0: i32):
    %c1 = arith.constant 1.0 : f32
    "test.types_compat_yield"(%c1) : (f32) -> ()
  }) : (i32) -> i32
  return %0 : i32
}
