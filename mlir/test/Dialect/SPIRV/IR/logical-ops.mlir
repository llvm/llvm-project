// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IEqual
//===----------------------------------------------------------------------===//

func @iequal_scalar(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK: spv.IEqual {{.*}}, {{.*}} : i32
  %0 = spv.IEqual %arg0, %arg1 : i32
  return %0 : i1
}

// -----

func @iequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.IEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.IEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.INotEqual
//===----------------------------------------------------------------------===//

func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.INotEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.IsInf
//===----------------------------------------------------------------------===//

func @isinf_scalar(%arg0: f32) -> i1 {
  // CHECK: spv.IsInf {{.*}} : f32
  %0 = spv.IsInf %arg0 : f32
  return %0 : i1
}

func @isinf_vector(%arg0: vector<2xf32>) -> vector<2xi1> {
  // CHECK: spv.IsInf {{.*}} : vector<2xf32>
  %0 = spv.IsInf %arg0 : vector<2xf32>
  return %0 : vector<2xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.IsNan
//===----------------------------------------------------------------------===//

func @isnan_scalar(%arg0: f32) -> i1 {
  // CHECK: spv.IsNan {{.*}} : f32
  %0 = spv.IsNan %arg0 : f32
  return %0 : i1
}

func @isnan_vector(%arg0: vector<2xf32>) -> vector<2xi1> {
  // CHECK: spv.IsNan {{.*}} : vector<2xf32>
  %0 = spv.IsNan %arg0 : vector<2xf32>
  return %0 : vector<2xi1>
}

//===----------------------------------------------------------------------===//
// spv.LogicalAnd
//===----------------------------------------------------------------------===//

func @logicalBinary(%arg0 : i1, %arg1 : i1, %arg2 : i1)
{
  // CHECK: [[TMP:%.*]] = spv.LogicalAnd {{%.*}}, {{%.*}} : i1
  %0 = spv.LogicalAnd %arg0, %arg1 : i1
  // CHECK: {{%.*}} = spv.LogicalAnd [[TMP]], {{%.*}} : i1
  %1 = spv.LogicalAnd %0, %arg2 : i1
  return
}

func @logicalBinary2(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>)
{
  // CHECK: {{%.*}} = spv.LogicalAnd {{%.*}}, {{%.*}} : vector<4xi1>
  %0 = spv.LogicalAnd %arg0, %arg1 : vector<4xi1>
  return
}

// -----

func @logicalBinary(%arg0 : i1, %arg1 : i1)
{
  // expected-error @+2 {{expected ':'}}
  %0 = spv.LogicalAnd %arg0, %arg1
  return
}

// -----

func @logicalBinary(%arg0 : i1, %arg1 : i1)
{
  // expected-error @+2 {{expected non-function type}}
  %0 = spv.LogicalAnd %arg0, %arg1 :
  return
}

// -----

func @logicalBinary(%arg0 : i1, %arg1 : i1)
{
  // expected-error @+1 {{custom op 'spv.LogicalAnd' expected 2 operands}}
  %0 = spv.LogicalAnd %arg0 : i1
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.LogicalNot
//===----------------------------------------------------------------------===//

func @logicalUnary(%arg0 : i1, %arg1 : i1)
{
  // CHECK: [[TMP:%.*]] = spv.LogicalNot {{%.*}} : i1
  %0 = spv.LogicalNot %arg0 : i1
  // CHECK: {{%.*}} = spv.LogicalNot [[TMP]] : i1
  %1 = spv.LogicalNot %0 : i1
  return
}

func @logicalUnary2(%arg0 : vector<4xi1>)
{
  // CHECK: {{%.*}} = spv.LogicalNot {{%.*}} : vector<4xi1>
  %0 = spv.LogicalNot %arg0 : vector<4xi1>
  return
}

// -----

func @logicalUnary(%arg0 : i1)
{
  // expected-error @+2 {{expected ':'}}
  %0 = spv.LogicalNot %arg0
  return
}

// -----

func @logicalUnary(%arg0 : i1)
{
  // expected-error @+2 {{expected non-function type}}
  %0 = spv.LogicalNot %arg0 :
  return
}

// -----

func @logicalUnary(%arg0 : i1)
{
  // expected-error @+1 {{expected SSA operand}}
  %0 = spv.LogicalNot : i1
  return
}

// -----

func @logicalUnary(%arg0 : i32)
{
  // expected-error @+1 {{operand #0 must be bool or vector of bool values of length 2/3/4/8/16, but got 'i32'}}
  %0 = spv.LogicalNot %arg0 : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.SelectOp
//===----------------------------------------------------------------------===//

func @select_op_bool(%arg0: i1) -> () {
  %0 = spv.Constant true
  %1 = spv.Constant false
  // CHECK : spv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, i1
  %2 = spv.Select %arg0, %0, %1 : i1, i1
  return
}

func @select_op_int(%arg0: i1) -> () {
  %0 = spv.Constant 2 : i32
  %1 = spv.Constant 3 : i32
  // CHECK : spv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, i32
  %2 = spv.Select %arg0, %0, %1 : i1, i32
  return
}

func @select_op_float(%arg0: i1) -> () {
  %0 = spv.Constant 2.0 : f32
  %1 = spv.Constant 3.0 : f32
  // CHECK : spv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, f32
  %2 = spv.Select %arg0, %0, %1 : i1, f32
  return
}

func @select_op_ptr(%arg0: i1) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  %1 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK : spv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, !spv.ptr<f32, Function>
  %2 = spv.Select %arg0, %0, %1 : i1, !spv.ptr<f32, Function>
  return
}

func @select_op_vec(%arg0: i1) -> () {
  %0 = spv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spv.Constant dense<[5.0, 6.0, 7.0]> : vector<3xf32>
  // CHECK : spv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, vector<3xf32>
  %2 = spv.Select %arg0, %0, %1 : i1, vector<3xf32>
  return
}

func @select_op_vec_condn_vec(%arg0: vector<3xi1>) -> () {
  %0 = spv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spv.Constant dense<[5.0, 6.0, 7.0]> : vector<3xf32>
  // CHECK : spv.Select {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi1>, vector<3xf32>
  %2 = spv.Select %arg0, %0, %1 : vector<3xi1>, vector<3xf32>
  return
}

// -----

func @select_op(%arg0: i1) -> () {
  %0 = spv.Constant 2 : i32
  %1 = spv.Constant 3 : i32
  // expected-error @+2 {{expected ','}}
  %2 = spv.Select %arg0, %0, %1 : i1
  return
}

// -----

func @select_op(%arg1: vector<3xi1>) -> () {
  %0 = spv.Constant 2 : i32
  %1 = spv.Constant 3 : i32
  // expected-error @+1 {{result expected to be of vector type when condition is of vector type}}
  %2 = spv.Select %arg1, %0, %1 : vector<3xi1>, i32
  return
}

// -----

func @select_op(%arg1: vector<4xi1>) -> () {
  %0 = spv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %1 = spv.Constant dense<[5, 6, 7]> : vector<3xi32>
  // expected-error @+1 {{result should have the same number of elements as the condition when condition is of vector type}}
  %2 = spv.Select %arg1, %0, %1 : vector<4xi1>, vector<3xi32>
  return
}

// -----

func @select_op(%arg1: vector<4xi1>) -> () {
  %0 = spv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spv.Constant dense<[5, 6, 7]> : vector<3xi32>
  // expected-error @+1 {{all of {true_value, false_value, result} have same type}}
  %2 = "spv.Select"(%arg1, %0, %1) : (vector<4xi1>, vector<3xf32>, vector<3xi32>) -> vector<3xi32>
  return
}

// -----

func @select_op(%arg1: vector<4xi1>) -> () {
  %0 = spv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spv.Constant dense<[5, 6, 7]> : vector<3xi32>
  // expected-error @+1 {{all of {true_value, false_value, result} have same type}}
  %2 = "spv.Select"(%arg1, %1, %0) : (vector<4xi1>, vector<3xi32>, vector<3xf32>) -> vector<3xi32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.SGreaterThan
//===----------------------------------------------------------------------===//

func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SLessThan
//===----------------------------------------------------------------------===//

func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SLessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SLessThanEqual
//===----------------------------------------------------------------------===//

func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.UGreaterThan
//===----------------------------------------------------------------------===//

func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.UGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ULessThan
//===----------------------------------------------------------------------===//

func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.ULessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ULessThanEqual
//===----------------------------------------------------------------------===//

func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}
