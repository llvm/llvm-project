// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func.func @test(%arg0 : !test.optional_value_type, %arg1 : !test.optional_value_type<3>) {
  // CHECK: test.format_maybe_empty_type %{{.*}} : !test.optional_value_type
  test.format_maybe_empty_type %arg0 : !test.optional_value_type
  // CHECK: test.format_maybe_empty_type %{{.*}} : <3>
  test.format_maybe_empty_type %arg1 : !test.optional_value_type<3>
  return
}
