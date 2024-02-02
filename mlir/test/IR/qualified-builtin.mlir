// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: @test1
// CHECK: -> tuple<>
func.func private @test1() -> !builtin.tuple<>

// CHECK-LABEL: @test2
// CHECK: -> none
func.func private @test2() -> !builtin.none


