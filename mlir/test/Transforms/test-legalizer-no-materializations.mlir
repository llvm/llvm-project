// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns="allow-pattern-rollback=0 build-materializations=0 attach-debug-materialization-kind=1" -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-KIND

// CHECK-LABEL: func @dropped_input_in_use
// CHECK-KIND-LABEL: func @dropped_input_in_use
func.func @dropped_input_in_use(%arg: i16, %arg2: i64) {
  // CHECK-NEXT: %[[cast:.*]] = "test.cast"() : () -> i16
  // CHECK-NEXT: "work"(%[[cast]]) : (i16)
  // CHECK-KIND-NEXT: %[[cast:.*]] = builtin.unrealized_conversion_cast to i16 {__kind__ = "source"}
  // CHECK-KIND-NEXT: "work"(%[[cast]]) : (i16)
  // expected-remark@+1 {{op 'work' is not legalizable}}
  "work"(%arg) : (i16) -> ()
}

// -----

// CHECK-KIND-LABEL: func @test_lookup_without_converter
//       CHECK-KIND:   %[[producer:.*]] = "test.valid_producer"() : () -> i16
//       CHECK-KIND:   %[[cast:.*]] = builtin.unrealized_conversion_cast %[[producer]] : i16 to f64 {__kind__ = "target"}
//       CHECK-KIND:   "test.valid_consumer"(%[[cast]]) : (f64) -> ()
//       CHECK-KIND:   "test.valid_consumer"(%[[producer]]) : (i16) -> ()
func.func @test_lookup_without_converter() {
  %0 = "test.replace_with_valid_producer"() {type = i16} : () -> (i64)
  "test.replace_with_valid_consumer"(%0) {with_converter} : (i64) -> ()
  // Make sure that the second "replace_with_valid_consumer" lowering does not
  // lookup the materialization that was created for the above op.
  "test.replace_with_valid_consumer"(%0) : (i64) -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

// CHECK-LABEL: func @remap_moved_region_args
func.func @remap_moved_region_args() {
  // CHECK-NEXT: return
  // CHECK-NEXT: ^bb1(%[[arg0:.*]]: i64, %[[arg1:.*]]: i16, %[[arg2:.*]]: i64, %[[arg3:.*]]: f32):
  // CHECK-NEXT: %[[cast1:.*]]:2 = builtin.unrealized_conversion_cast %[[arg3]] : f32 to f16, f16
  // CHECK-NEXT: %[[cast2:.*]] = builtin.unrealized_conversion_cast %[[arg2]] : i64 to f64
  // CHECK-NEXT: %[[cast3:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to f64
  // CHECK-NEXT: %[[cast4:.*]] = "test.cast"(%[[cast1]]#0, %[[cast1]]#1) : (f16, f16) -> f32
  // CHECK-NEXT: "test.valid"(%[[cast3]], %[[cast2]], %[[cast4]]) : (f64, f64, f32)
  "test.region"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

// CHECK-LABEL: func @remap_cloned_region_args
func.func @remap_cloned_region_args() {
  // CHECK-NEXT: return
  // CHECK-NEXT: ^bb1(%[[arg0:.*]]: i64, %[[arg1:.*]]: i16, %[[arg2:.*]]: i64, %[[arg3:.*]]: f32):
  // CHECK-NEXT: %[[cast1:.*]]:2 = builtin.unrealized_conversion_cast %[[arg3]] : f32 to f16, f16
  // CHECK-NEXT: %[[cast2:.*]] = builtin.unrealized_conversion_cast %[[arg2]] : i64 to f64
  // CHECK-NEXT: %[[cast3:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to f64
  // CHECK-NEXT: %[[cast4:.*]] = "test.cast"(%[[cast1]]#0, %[[cast1]]#1) : (f16, f16) -> f32
  // CHECK-NEXT: "test.valid"(%[[cast3]], %[[cast2]], %[[cast4]]) : (f64, f64, f32)
  "test.region"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) {legalizer.should_clone} : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}
