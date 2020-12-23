// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformBallot
//===----------------------------------------------------------------------===//

func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: %{{.*}} = spv.GroupNonUniformBallot "Workgroup" %{{.*}}: vector<4xi32>
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// -----

func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spv.GroupNonUniformBallot "Device" %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// -----

func @group_non_uniform_ballot(%predicate: i1) -> vector<4xsi32> {
  // expected-error @+1 {{op result #0 must be vector of 8/16/32/64-bit signless/unsigned integer values of length 4, but got 'vector<4xsi32>'}}
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xsi32>
  return %0: vector<4xsi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.NonUniformGroupBroadcast
//===----------------------------------------------------------------------===//

func @group_non_uniform_broadcast_scalar(%value: f32) -> f32 {
  %one = spv.constant 1 : i32
  // CHECK: spv.GroupNonUniformBroadcast "Workgroup" %{{.*}}, %{{.*}} : f32, i32
  %0 = spv.GroupNonUniformBroadcast "Workgroup" %value, %one : f32, i32
  return %0: f32
}

// -----

func @group_non_uniform_broadcast_vector(%value: vector<4xf32>) -> vector<4xf32> {
  %one = spv.constant 1 : i32
  // CHECK: spv.GroupNonUniformBroadcast "Subgroup" %{{.*}}, %{{.*}} : vector<4xf32>, i32
  %0 = spv.GroupNonUniformBroadcast "Subgroup" %value, %one : vector<4xf32>, i32
  return %0: vector<4xf32>
}

// -----

func @group_non_uniform_broadcast_negative_scope(%value: f32, %localid: i32 ) -> f32 {
  %one = spv.constant 1 : i32
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}} 
  %0 = spv.GroupNonUniformBroadcast "Device" %value, %one : f32, i32
  return %0: f32
}

// -----

func @group_non_uniform_broadcast_negative_non_const(%value: f32, %localid: i32) -> f32 {
  // expected-error @+1 {{id must be the result of a constant op}}
  %0 = spv.GroupNonUniformBroadcast "Subgroup" %value, %localid : f32, i32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformElect
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_elect
func @group_non_uniform_elect() -> i1 {
  // CHECK: %{{.+}} = spv.GroupNonUniformElect "Workgroup" : i1
  %0 = spv.GroupNonUniformElect "Workgroup" : i1
  return %0: i1
}

// -----

func @group_non_uniform_elect() -> i1 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spv.GroupNonUniformElect "CrossDevice" : i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fadd_reduce
func @group_non_uniform_fadd_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %{{.+}} : f32
  %0 = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_fadd_clustered_reduce
func @group_non_uniform_fadd_clustered_reduce(%val: vector<2xf32>) -> vector<2xf32> {
  %four = spv.constant 4 : i32
  // CHECK: %{{.+}} = spv.GroupNonUniformFAdd "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xf32>
  %0 = spv.GroupNonUniformFAdd "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xf32>
  return %0: vector<2xf32>
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fmul_reduce
func @group_non_uniform_fmul_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformFMul "Workgroup" "Reduce" %{{.+}} : f32
  %0 = spv.GroupNonUniformFMul "Workgroup" "Reduce" %val : f32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_fmul_clustered_reduce
func @group_non_uniform_fmul_clustered_reduce(%val: vector<2xf32>) -> vector<2xf32> {
  %four = spv.constant 4 : i32
  // CHECK: %{{.+}} = spv.GroupNonUniformFMul "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xf32>
  %0 = spv.GroupNonUniformFMul "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xf32>
  return %0: vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fmax_reduce
func @group_non_uniform_fmax_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformFMax "Workgroup" "Reduce" %{{.+}} : f32
  %0 = spv.GroupNonUniformFMax "Workgroup" "Reduce" %val : f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fmin_reduce
func @group_non_uniform_fmin_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformFMin "Workgroup" "Reduce" %{{.+}} : f32
  %0 = spv.GroupNonUniformFMin "Workgroup" "Reduce" %val : f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformIAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_iadd_reduce
func @group_non_uniform_iadd_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %{{.+}} : i32
  %0 = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %val : i32
  return %0: i32
}

// CHECK-LABEL: @group_non_uniform_iadd_clustered_reduce
func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  %four = spv.constant 4 : i32
  // CHECK: %{{.+}} = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>
  %0 = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xi32>
  return %0: vector<2xi32>
}

// -----

func @group_non_uniform_iadd_reduce(%val: i32) -> i32 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spv.GroupNonUniformIAdd "Device" "Reduce" %val : i32
  return %0: i32
}

// -----

func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  // expected-error @+1 {{cluster size operand must be provided for 'ClusteredReduce' group operation}}
  %0 = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val : vector<2xi32>
  return %0: vector<2xi32>
}

// -----

func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>, %size: i32) -> vector<2xi32> {
  // expected-error @+1 {{cluster size operand must come from a constant op}}
  %0 = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val cluster_size(%size) : vector<2xi32>
  return %0: vector<2xi32>
}

// -----

func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  %five = spv.constant 5 : i32
  // expected-error @+1 {{cluster size operand must be a power of two}}
  %0 = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val cluster_size(%five) : vector<2xi32>
  return %0: vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformIMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_imul_reduce
func @group_non_uniform_imul_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformIMul "Workgroup" "Reduce" %{{.+}} : i32
  %0 = spv.GroupNonUniformIMul "Workgroup" "Reduce" %val : i32
  return %0: i32
}

// CHECK-LABEL: @group_non_uniform_imul_clustered_reduce
func @group_non_uniform_imul_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  %four = spv.constant 4 : i32
  // CHECK: %{{.+}} = spv.GroupNonUniformIMul "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>
  %0 = spv.GroupNonUniformIMul "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xi32>
  return %0: vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformSMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_smax_reduce
func @group_non_uniform_smax_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformSMax "Workgroup" "Reduce" %{{.+}} : i32
  %0 = spv.GroupNonUniformSMax "Workgroup" "Reduce" %val : i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformSMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_smin_reduce
func @group_non_uniform_smin_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformSMin "Workgroup" "Reduce" %{{.+}} : i32
  %0 = spv.GroupNonUniformSMin "Workgroup" "Reduce" %val : i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformUMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_umax_reduce
func @group_non_uniform_umax_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformUMax "Workgroup" "Reduce" %{{.+}} : i32
  %0 = spv.GroupNonUniformUMax "Workgroup" "Reduce" %val : i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformUMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_umin_reduce
func @group_non_uniform_umin_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spv.GroupNonUniformUMin "Workgroup" "Reduce" %{{.+}} : i32
  %0 = spv.GroupNonUniformUMin "Workgroup" "Reduce" %val : i32
  return %0: i32
}
