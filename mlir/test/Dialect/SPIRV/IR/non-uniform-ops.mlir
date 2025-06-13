// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBallot
//===----------------------------------------------------------------------===//

func.func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: %{{.*}} = spirv.GroupNonUniformBallot <Workgroup> %{{.*}}: vector<4xi32>
  %0 = spirv.GroupNonUniformBallot <Workgroup> %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// -----

func.func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformBallot <Device> %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// -----

func.func @group_non_uniform_ballot(%predicate: i1) -> vector<4xsi32> {
  // expected-error @+1 {{op result #0 must be vector of 8/16/32/64-bit signless/unsigned integer values of length 4, but got 'vector<4xsi32>'}}
  %0 = spirv.GroupNonUniformBallot <Workgroup> %predicate : vector<4xsi32>
  return %0: vector<4xsi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBallotFindLSB
//===----------------------------------------------------------------------===//

func.func @group_non_uniform_ballot_find_lsb(%value : vector<4xi32>) -> i32 {
  // CHECK: %{{.*}} = spirv.GroupNonUniformBallotFindLSB <Subgroup> %{{.*}}: vector<4xi32>, i32
  %0 = spirv.GroupNonUniformBallotFindLSB <Subgroup> %value : vector<4xi32>, i32
  return %0: i32
}

// -----

func.func @group_non_uniform_ballot_find_lsb(%value : vector<4xi32>) -> i32 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformBallotFindLSB <Device> %value : vector<4xi32>, i32
  return %0: i32
}

// -----

func.func @group_non_uniform_ballot_find_lsb(%value : vector<4xi32>) -> si32 {
  // expected-error @+1 {{op result #0 must be 8/16/32/64-bit signless/unsigned integer, but got 'si32'}}
  %0 = spirv.GroupNonUniformBallotFindLSB <Subgroup> %value : vector<4xi32>, si32
  return %0: si32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBallotFindLSB
//===----------------------------------------------------------------------===//

func.func @group_non_uniform_ballot_find_msb(%value : vector<4xi32>) -> i32 {
  // CHECK: %{{.*}} = spirv.GroupNonUniformBallotFindMSB <Subgroup> %{{.*}}: vector<4xi32>, i32
  %0 = spirv.GroupNonUniformBallotFindMSB <Subgroup> %value : vector<4xi32>, i32
  return %0: i32
}

// -----

func.func @group_non_uniform_ballot_find_msb(%value : vector<4xi32>) -> i32 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformBallotFindMSB <Device> %value : vector<4xi32>, i32
  return %0: i32
}

// -----

func.func @group_non_uniform_ballot_find_msb(%value : vector<4xi32>) -> si32 {
  // expected-error @+1 {{op result #0 must be 8/16/32/64-bit signless/unsigned integer, but got 'si32'}}
  %0 = spirv.GroupNonUniformBallotFindMSB <Subgroup> %value : vector<4xi32>, si32
  return %0: si32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.NonUniformGroupBroadcast
//===----------------------------------------------------------------------===//

func.func @group_non_uniform_broadcast_scalar(%value: f32) -> f32 {
  %one = spirv.Constant 1 : i32
  // CHECK: spirv.GroupNonUniformBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, i32
  %0 = spirv.GroupNonUniformBroadcast <Workgroup> %value, %one : f32, i32
  return %0: f32
}

// -----

func.func @group_non_uniform_broadcast_vector(%value: vector<4xf32>) -> vector<4xf32> {
  %one = spirv.Constant 1 : i32
  // CHECK: spirv.GroupNonUniformBroadcast <Subgroup> %{{.*}}, %{{.*}} : vector<4xf32>, i32
  %0 = spirv.GroupNonUniformBroadcast <Subgroup> %value, %one : vector<4xf32>, i32
  return %0: vector<4xf32>
}

// -----

func.func @group_non_uniform_broadcast_negative_scope(%value: f32, %localid: i32 ) -> f32 {
  %one = spirv.Constant 1 : i32
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformBroadcast <Device> %value, %one : f32, i32
  return %0: f32
}

// -----

func.func @group_non_uniform_broadcast_negative_non_const(%value: f32, %localid: i32) -> f32 {
  // expected-error @+1 {{id must be the result of a constant op}}
  %0 = spirv.GroupNonUniformBroadcast <Subgroup> %value, %localid : f32, i32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformElect
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_elect
func.func @group_non_uniform_elect() -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformElect <Workgroup> : i1
  %0 = spirv.GroupNonUniformElect <Workgroup> : i1
  return %0: i1
}

// -----

func.func @group_non_uniform_elect() -> i1 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformElect <CrossDevice> : i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fadd_reduce
func.func @group_non_uniform_fadd_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformFAdd <Workgroup> <Reduce> %{{.+}} : f32 -> f32
  %0 = spirv.GroupNonUniformFAdd <Workgroup> <Reduce> %val : f32 -> f32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_fadd_clustered_reduce
func.func @group_non_uniform_fadd_clustered_reduce(%val: vector<2xf32>) -> vector<2xf32> {
  %four = spirv.Constant 4 : i32
  // CHECK: %{{.+}} = spirv.GroupNonUniformFAdd <Workgroup> <ClusteredReduce> %{{.+}} cluster_size(%{{.+}}) : vector<2xf32>, i32 -> vector<2xf32>
  %0 = spirv.GroupNonUniformFAdd <Workgroup> <ClusteredReduce> %val cluster_size(%four) : vector<2xf32>, i32 -> vector<2xf32>
  return %0: vector<2xf32>
}

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fmul_reduce
func.func @group_non_uniform_fmul_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformFMul <Workgroup> <Reduce> %{{.+}} : f32 -> f32
  %0 = spirv.GroupNonUniformFMul <Workgroup> <Reduce> %val : f32 -> f32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_fmul_clustered_reduce
func.func @group_non_uniform_fmul_clustered_reduce(%val: vector<2xf32>) -> vector<2xf32> {
  %four = spirv.Constant 4 : i32
  // CHECK: %{{.+}} = spirv.GroupNonUniformFMul <Workgroup> <ClusteredReduce> %{{.+}} cluster_size(%{{.+}}) : vector<2xf32>, i32 -> vector<2xf32>
  %0 = spirv.GroupNonUniformFMul <Workgroup> <ClusteredReduce> %val cluster_size(%four) : vector<2xf32>, i32 -> vector<2xf32>
  return %0: vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fmax_reduce
func.func @group_non_uniform_fmax_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformFMax <Workgroup> <Reduce> %{{.+}} : f32 -> f32
  %0 = spirv.GroupNonUniformFMax <Workgroup> <Reduce> %val : f32 -> f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformFMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_fmin_reduce
func.func @group_non_uniform_fmin_reduce(%val: f32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformFMin <Workgroup> <Reduce> %{{.+}} : f32 -> f32
  %0 = spirv.GroupNonUniformFMin <Workgroup> <Reduce> %val : f32 -> f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformIAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_iadd_reduce
func.func @group_non_uniform_iadd_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformIAdd <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformIAdd <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// CHECK-LABEL: @group_non_uniform_iadd_clustered_reduce
func.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  %four = spirv.Constant 4 : i32
  // CHECK: %{{.+}} = spirv.GroupNonUniformIAdd <Workgroup> <ClusteredReduce> %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>, i32 -> vector<2xi32>
  %0 = spirv.GroupNonUniformIAdd <Workgroup> <ClusteredReduce> %val cluster_size(%four) : vector<2xi32>, i32 -> vector<2xi32>
  return %0: vector<2xi32>
}

// -----

func.func @group_non_uniform_iadd_reduce(%val: i32) -> i32 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformIAdd <Device> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

func.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  // expected-error @+1 {{cluster size operand must be provided for 'ClusteredReduce' group operation}}
  %0 = spirv.GroupNonUniformIAdd <Workgroup> <ClusteredReduce> %val : vector<2xi32> -> vector<2xi32>
  return %0: vector<2xi32>
}

// -----

func.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>, %size: i32) -> vector<2xi32> {
  // expected-error @+1 {{cluster size operand must come from a constant op}}
  %0 = spirv.GroupNonUniformIAdd <Workgroup> <ClusteredReduce> %val cluster_size(%size) : vector<2xi32>, i32 -> vector<2xi32>
  return %0: vector<2xi32>
}

// -----

func.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  %five = spirv.Constant 5 : i32
  // expected-error @+1 {{cluster size operand must be a power of two}}
  %0 = spirv.GroupNonUniformIAdd <Workgroup> <ClusteredReduce> %val cluster_size(%five) : vector<2xi32>, i32 -> vector<2xi32>
  return %0: vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformIMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_imul_reduce
func.func @group_non_uniform_imul_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformIMul <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformIMul <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// CHECK-LABEL: @group_non_uniform_imul_clustered_reduce
func.func @group_non_uniform_imul_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
  %four = spirv.Constant 4 : i32
  // CHECK: %{{.+}} = spirv.GroupNonUniformIMul <Workgroup> <ClusteredReduce> %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>, i32 -> vector<2xi32>
  %0 = spirv.GroupNonUniformIMul <Workgroup> <ClusteredReduce> %val cluster_size(%four) : vector<2xi32>, i32 -> vector<2xi32>
  return %0: vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformSMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_smax_reduce
func.func @group_non_uniform_smax_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformSMax <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformSMax <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformSMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_smin_reduce
func.func @group_non_uniform_smin_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformSMin <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformSMin <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformShuffle
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_shuffle1
func.func @group_non_uniform_shuffle1(%val: f32, %id: i32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffle <Subgroup> %{{.+}}, %{{.+}} : f32, i32
  %0 = spirv.GroupNonUniformShuffle <Subgroup> %val, %id : f32, i32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_shuffle2
func.func @group_non_uniform_shuffle2(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffle <Subgroup> %{{.+}}, %{{.+}} : vector<2xf32>, i32
  %0 = spirv.GroupNonUniformShuffle <Subgroup> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformShuffle <Device> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: si32) -> vector<2xf32> {
  // expected-error @+1 {{second operand must be a singless/unsigned integer}}
  %0 = spirv.GroupNonUniformShuffle <Subgroup> %val, %id : vector<2xf32>, si32
  return %0: vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformShuffleXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_shuffle1
func.func @group_non_uniform_shuffle1(%val: f32, %id: i32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleXor <Subgroup> %{{.+}}, %{{.+}} : f32, i32
  %0 = spirv.GroupNonUniformShuffleXor <Subgroup> %val, %id : f32, i32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_shuffle2
func.func @group_non_uniform_shuffle2(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleXor <Subgroup> %{{.+}}, %{{.+}} : vector<2xf32>, i32
  %0 = spirv.GroupNonUniformShuffleXor <Subgroup> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformShuffleXor <Device> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: si32) -> vector<2xf32> {
  // expected-error @+1 {{second operand must be a singless/unsigned integer}}
  %0 = spirv.GroupNonUniformShuffleXor <Subgroup> %val, %id : vector<2xf32>, si32
  return %0: vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformShuffleUp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_shuffle1
func.func @group_non_uniform_shuffle1(%val: f32, %id: i32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleUp <Subgroup> %{{.+}}, %{{.+}} : f32, i32
  %0 = spirv.GroupNonUniformShuffleUp <Subgroup> %val, %id : f32, i32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_shuffle2
func.func @group_non_uniform_shuffle2(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleUp <Subgroup> %{{.+}}, %{{.+}} : vector<2xf32>, i32
  %0 = spirv.GroupNonUniformShuffleUp <Subgroup> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformShuffleUp <Device> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: si32) -> vector<2xf32> {
  // expected-error @+1 {{second operand must be a singless/unsigned integer}}
  %0 = spirv.GroupNonUniformShuffleUp <Subgroup> %val, %id : vector<2xf32>, si32
  return %0: vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformShuffleDown
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_shuffle1
func.func @group_non_uniform_shuffle1(%val: f32, %id: i32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleDown <Subgroup> %{{.+}}, %{{.+}} : f32, i32
  %0 = spirv.GroupNonUniformShuffleDown <Subgroup> %val, %id : f32, i32
  return %0: f32
}

// CHECK-LABEL: @group_non_uniform_shuffle2
func.func @group_non_uniform_shuffle2(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleDown <Subgroup> %{{.+}}, %{{.+}} : vector<2xf32>, i32
  %0 = spirv.GroupNonUniformShuffleDown <Subgroup> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: i32) -> vector<2xf32> {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformShuffleDown <Device> %val, %id : vector<2xf32>, i32
  return %0: vector<2xf32>
}

// -----

func.func @group_non_uniform_shuffle(%val: vector<2xf32>, %id: si32) -> vector<2xf32> {
  // expected-error @+1 {{second operand must be a singless/unsigned integer}}
  %0 = spirv.GroupNonUniformShuffleDown <Subgroup> %val, %id : vector<2xf32>, si32
  return %0: vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformUMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_umax_reduce
func.func @group_non_uniform_umax_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformUMax <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformUMax <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformUMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_umin_reduce
func.func @group_non_uniform_umin_reduce(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformUMin <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformUMin <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_bitwise_and
func.func @group_non_uniform_bitwise_and(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformBitwiseAnd <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformBitwiseAnd <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

func.func @group_non_uniform_bitwise_and(%val: i1) -> i1 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16, but got 'i1'}}
  %0 = spirv.GroupNonUniformBitwiseAnd <Workgroup> <Reduce> %val : i1 -> i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_bitwise_or
func.func @group_non_uniform_bitwise_or(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformBitwiseOr <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformBitwiseOr <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

func.func @group_non_uniform_bitwise_or(%val: i1) -> i1 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16, but got 'i1'}}
  %0 = spirv.GroupNonUniformBitwiseOr <Workgroup> <Reduce> %val : i1 -> i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformBitwiseXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_bitwise_xor
func.func @group_non_uniform_bitwise_xor(%val: i32) -> i32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformBitwiseXor <Workgroup> <Reduce> %{{.+}} : i32 -> i32
  %0 = spirv.GroupNonUniformBitwiseXor <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

func.func @group_non_uniform_bitwise_xor(%val: i1) -> i1 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16, but got 'i1'}}
  %0 = spirv.GroupNonUniformBitwiseXor <Workgroup> <Reduce> %val : i1 -> i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformLogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_logical_and
func.func @group_non_uniform_logical_and(%val: i1) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformLogicalAnd <Workgroup> <Reduce> %{{.+}} : i1 -> i1
  %0 = spirv.GroupNonUniformLogicalAnd <Workgroup> <Reduce> %val : i1 -> i1
  return %0: i1
}

// -----

func.func @group_non_uniform_logical_and(%val: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be bool or vector of bool values of length 2/3/4/8/16, but got 'i32'}}
  %0 = spirv.GroupNonUniformLogicalAnd <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformLogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_logical_or
func.func @group_non_uniform_logical_or(%val: i1) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformLogicalOr <Workgroup> <Reduce> %{{.+}} : i1 -> i1
  %0 = spirv.GroupNonUniformLogicalOr <Workgroup> <Reduce> %val : i1 -> i1
  return %0: i1
}

// -----

func.func @group_non_uniform_logical_or(%val: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be bool or vector of bool values of length 2/3/4/8/16, but got 'i32'}}
  %0 = spirv.GroupNonUniformLogicalOr <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformLogicalXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_logical_xor
func.func @group_non_uniform_logical_xor(%val: i1) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformLogicalXor <Workgroup> <Reduce> %{{.+}} : i1 -> i1
  %0 = spirv.GroupNonUniformLogicalXor <Workgroup> <Reduce> %val : i1 -> i1
  return %0: i1
}

// -----

func.func @group_non_uniform_logical_xor(%val: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be bool or vector of bool values of length 2/3/4/8/16, but got 'i32'}}
  %0 = spirv.GroupNonUniformLogicalXor <Workgroup> <Reduce> %val : i32 -> i32
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformRotateKHR
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_rotate_khr
func.func @group_non_uniform_rotate_khr(%val: f32, %delta: i32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformRotateKHR <Subgroup> %{{.+}} : f32, i32 -> f32
  %0 = spirv.GroupNonUniformRotateKHR <Subgroup> %val, %delta : f32, i32 -> f32
  return %0: f32
}

// -----

// CHECK-LABEL: @group_non_uniform_rotate_khr
func.func @group_non_uniform_rotate_khr(%val: f32, %delta: i32) -> f32 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformRotateKHR <Workgroup> %{{.+}} : f32, i32, i32 -> f32
  %four = spirv.Constant 4 : i32
  %0 = spirv.GroupNonUniformRotateKHR <Workgroup> %val, %delta, cluster_size(%four) : f32, i32, i32 -> f32
  return %0: f32
}

// -----

func.func @group_non_uniform_rotate_khr(%val: f32, %delta: i32) -> f32 {
  %four = spirv.Constant 4 : i32
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}}
  %0 = spirv.GroupNonUniformRotateKHR <Device> %val, %delta, cluster_size(%four) : f32, i32, i32 -> f32
  return %0: f32
}

// -----

func.func @group_non_uniform_rotate_khr(%val: f32, %delta: si32) -> f32 {
  %four = spirv.Constant 4 : i32
  // expected-error @+1 {{op operand #1 must be 8/16/32/64-bit signless/unsigned integer, but got 'si32'}}
  %0 = spirv.GroupNonUniformRotateKHR <Subgroup> %val, %delta, cluster_size(%four) : f32, si32, i32 -> f32
  return %0: f32
}

// -----

func.func @group_non_uniform_rotate_khr(%val: f32, %delta: i32) -> f32 {
  %four = spirv.Constant 4 : si32
  // expected-error @+1 {{op operand #2 must be 8/16/32/64-bit signless/unsigned integer, but got 'si32'}}
  %0 = spirv.GroupNonUniformRotateKHR <Subgroup> %val, %delta, cluster_size(%four) : f32, i32, si32 -> f32
  return %0: f32
}

// -----

func.func @group_non_uniform_rotate_khr(%val: f32, %delta: i32, %four: i32) -> f32 {
  // expected-error @+1 {{cluster size operand must come from a constant op}}
  %0 = spirv.GroupNonUniformRotateKHR <Subgroup> %val, %delta, cluster_size(%four) : f32, i32, i32 -> f32
  return %0: f32
}

// -----

func.func @group_non_uniform_rotate_khr(%val: f32, %delta: i32) -> f32 {
  %five = spirv.Constant 5 : i32
  // expected-error @+1 {{cluster size operand must be a power of two}}
  %0 = spirv.GroupNonUniformRotateKHR <Subgroup> %val, %delta, cluster_size(%five) : f32, i32, i32 -> f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformAll
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_all
func.func @group_non_uniform_all(%predicate: i1) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformAll <Subgroup> %{{.+}} : i1
  %0 = spirv.GroupNonUniformAll <Subgroup> %predicate : i1
  return %0: i1
}

// -----

func.func @group_non_uniform_all(%predicate: i1) -> i1 {
  // expected-error @+1 {{execution_scope must be Scope of value Subgroup}}
  %0 = spirv.GroupNonUniformAll <Device> %predicate : i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformAny
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_any
func.func @group_non_uniform_any(%predicate: i1) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformAny <Subgroup> %{{.+}} : i1
  %0 = spirv.GroupNonUniformAny <Subgroup> %predicate : i1
  return %0: i1
}

// -----

func.func @group_non_uniform_any(%predicate: i1) -> i1 {
  // expected-error @+1 {{execution_scope must be Scope of value Subgroup}}
  %0 = spirv.GroupNonUniformAny <Device> %predicate : i1
  return %0: i1
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupNonUniformAllEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_all_equal
func.func @group_non_uniform_all_equal(%value: f32) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformAllEqual <Subgroup> %{{.+}} : f32, i1
  %0 = spirv.GroupNonUniformAllEqual <Subgroup> %value : f32, i1
  return %0: i1
}

// -----

// CHECK-LABEL: @group_non_uniform_all_equal
func.func @group_non_uniform_all_equal(%value: vector<4xi32>) -> i1 {
  // CHECK: %{{.+}} = spirv.GroupNonUniformAllEqual <Subgroup> %{{.+}} : vector<4xi32>, i1
  %0 = spirv.GroupNonUniformAllEqual <Subgroup> %value : vector<4xi32>, i1
  return %0: i1
}


// -----

func.func @group_non_uniform_all_equal(%value: f32) -> i1 {
  // expected-error @+1 {{execution_scope must be Scope of value Subgroup}}
  %0 = spirv.GroupNonUniformAllEqual <Device> %value : f32, i1
  return %0: i1
}
