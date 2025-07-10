// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.IEqual
//===----------------------------------------------------------------------===//

func.func @iequal_scalar(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK: spirv.IEqual {{.*}}, {{.*}} : i32
  %0 = spirv.IEqual %arg0, %arg1 : i32
  return %0 : i1
}

// -----

func.func @iequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.IEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.IEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INotEqual
//===----------------------------------------------------------------------===//

func.func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.INotEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IsInf
//===----------------------------------------------------------------------===//

func.func @isinf_scalar(%arg0: f32) -> i1 {
  // CHECK: spirv.IsInf {{.*}} : f32
  %0 = spirv.IsInf %arg0 : f32
  return %0 : i1
}

func.func @isinf_vector(%arg0: vector<2xf32>) -> vector<2xi1> {
  // CHECK: spirv.IsInf {{.*}} : vector<2xf32>
  %0 = spirv.IsInf %arg0 : vector<2xf32>
  return %0 : vector<2xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IsNan
//===----------------------------------------------------------------------===//

func.func @isnan_scalar(%arg0: f32) -> i1 {
  // CHECK: spirv.IsNan {{.*}} : f32
  %0 = spirv.IsNan %arg0 : f32
  return %0 : i1
}

func.func @isnan_vector(%arg0: vector<2xf32>) -> vector<2xi1> {
  // CHECK: spirv.IsNan {{.*}} : vector<2xf32>
  %0 = spirv.IsNan %arg0 : vector<2xf32>
  return %0 : vector<2xi1>
}

//===----------------------------------------------------------------------===//
// spirv.LogicalAnd
//===----------------------------------------------------------------------===//

func.func @logicalBinary(%arg0 : i1, %arg1 : i1, %arg2 : i1)
{
  // CHECK: [[TMP:%.*]] = spirv.LogicalAnd {{%.*}}, {{%.*}} : i1
  %0 = spirv.LogicalAnd %arg0, %arg1 : i1
  // CHECK: {{%.*}} = spirv.LogicalAnd [[TMP]], {{%.*}} : i1
  %1 = spirv.LogicalAnd %0, %arg2 : i1
  return
}

func.func @logicalBinary2(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>)
{
  // CHECK: {{%.*}} = spirv.LogicalAnd {{%.*}}, {{%.*}} : vector<4xi1>
  %0 = spirv.LogicalAnd %arg0, %arg1 : vector<4xi1>
  return
}

// -----

func.func @logicalBinary(%arg0 : i1, %arg1 : i1)
{
  // expected-error @+1 {{expected ':'}}
  %0 = spirv.LogicalAnd %arg0, %arg1
  return
}

// -----

func.func @logicalBinary(%arg0 : i1, %arg1 : i1)
{
  // expected-error @+1 {{expected non-function type}}
  %0 = spirv.LogicalAnd %arg0, %arg1 :
  return
}

// -----

func.func @logicalBinary(%arg0 : i1, %arg1 : i1)
{
  // expected-error @+1 {{expected ','}}
  %0 = spirv.LogicalAnd %arg0 : i1
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LogicalNot
//===----------------------------------------------------------------------===//

func.func @logicalUnary(%arg0 : i1, %arg1 : i1)
{
  // CHECK: [[TMP:%.*]] = spirv.LogicalNot {{%.*}} : i1
  %0 = spirv.LogicalNot %arg0 : i1
  // CHECK: {{%.*}} = spirv.LogicalNot [[TMP]] : i1
  %1 = spirv.LogicalNot %0 : i1
  return
}

func.func @logicalUnary2(%arg0 : vector<4xi1>)
{
  // CHECK: {{%.*}} = spirv.LogicalNot {{%.*}} : vector<4xi1>
  %0 = spirv.LogicalNot %arg0 : vector<4xi1>
  return
}

// -----

func.func @logicalUnary(%arg0 : i1)
{
  // expected-error @+1 {{expected ':'}}
  %0 = spirv.LogicalNot %arg0
  return
}

// -----

func.func @logicalUnary(%arg0 : i1)
{
  // expected-error @+1 {{expected non-function type}}
  %0 = spirv.LogicalNot %arg0 :
  return
}

// -----

func.func @logicalUnary(%arg0 : i1)
{
  // expected-error @+1 {{expected SSA operand}}
  %0 = spirv.LogicalNot : i1
  return
}

// -----

func.func @logicalUnary(%arg0 : i32)
{
  // expected-error @+1 {{'operand' must be bool or vector of bool values of length 2/3/4/8/16, but got 'i32'}}
  %0 = spirv.LogicalNot %arg0 : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SelectOp
//===----------------------------------------------------------------------===//

func.func @select_op_bool(%arg0: i1) -> () {
  %0 = spirv.Constant true
  %1 = spirv.Constant false
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, i1
  %2 = spirv.Select %arg0, %0, %1 : i1, i1
  return
}

func.func @select_op_int(%arg0: i1) -> () {
  %0 = spirv.Constant 2 : i32
  %1 = spirv.Constant 3 : i32
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, i32
  %2 = spirv.Select %arg0, %0, %1 : i1, i32
  return
}

func.func @select_op_float(%arg0: i1) -> () {
  %0 = spirv.Constant 2.0 : f32
  %1 = spirv.Constant 3.0 : f32
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, f32
  %2 = spirv.Select %arg0, %0, %1 : i1, f32
  return
}

func.func @select_op_bfloat16(%arg0: i1) -> () {
  %0 = spirv.Constant 2.0 : bf16
  %1 = spirv.Constant 3.0 : bf16
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, bf16
  %2 = spirv.Select %arg0, %0, %1 : i1, bf16
  return
}

func.func @select_op_ptr(%arg0: i1) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  %1 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, !spirv.ptr<f32, Function>
  %2 = spirv.Select %arg0, %0, %1 : i1, !spirv.ptr<f32, Function>
  return
}

func.func @select_op_vec(%arg0: i1) -> () {
  %0 = spirv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spirv.Constant dense<[5.0, 6.0, 7.0]> : vector<3xf32>
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : i1, vector<3xf32>
  %2 = spirv.Select %arg0, %0, %1 : i1, vector<3xf32>
  return
}

func.func @select_op_vec_condn_vec(%arg0: vector<3xi1>) -> () {
  %0 = spirv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spirv.Constant dense<[5.0, 6.0, 7.0]> : vector<3xf32>
  // CHECK: spirv.Select {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi1>, vector<3xf32>
  %2 = spirv.Select %arg0, %0, %1 : vector<3xi1>, vector<3xf32>
  return
}

// -----

func.func @select_op(%arg0: i1) -> () {
  %0 = spirv.Constant 2 : i32
  %1 = spirv.Constant 3 : i32
  // expected-error @+1 {{expected ','}}
  %2 = spirv.Select %arg0, %0, %1 : i1
  return
}

// -----

func.func @select_op(%arg1: vector<3xi1>) -> () {
  %0 = spirv.Constant 2 : i32
  %1 = spirv.Constant 3 : i32
  // expected-error @+1 {{result expected to be of vector type when condition is of vector type}}
  %2 = spirv.Select %arg1, %0, %1 : vector<3xi1>, i32
  return
}

// -----

func.func @select_op(%arg1: vector<4xi1>) -> () {
  %0 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %1 = spirv.Constant dense<[5, 6, 7]> : vector<3xi32>
  // expected-error @+1 {{result should have the same number of elements as the condition when condition is of vector type}}
  %2 = spirv.Select %arg1, %0, %1 : vector<4xi1>, vector<3xi32>
  return
}

// -----

func.func @select_op(%arg1: vector<4xi1>) -> () {
  %0 = spirv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spirv.Constant dense<[5, 6, 7]> : vector<3xi32>
  // expected-error @+1 {{all of {true_value, false_value, result} have same type}}
  %2 = "spirv.Select"(%arg1, %0, %1) : (vector<4xi1>, vector<3xf32>, vector<3xi32>) -> vector<3xi32>
  return
}

// -----

func.func @select_op(%arg1: vector<4xi1>) -> () {
  %0 = spirv.Constant dense<[2.0, 3.0, 4.0]> : vector<3xf32>
  %1 = spirv.Constant dense<[5, 6, 7]> : vector<3xi32>
  // TODO: expand post change in verification order. This is currently only
  // verifying that the type verification is failing but not the specific error
  // message. In final state the error should refer to mismatch in true_value and
  // false_value.
  // expected-error @+1 {{type}}
  %2 = "spirv.Select"(%arg1, %1, %0) : (vector<4xi1>, vector<3xi32>, vector<3xf32>) -> vector<3xi32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SGreaterThan
//===----------------------------------------------------------------------===//

func.func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.SGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

func.func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SLessThan
//===----------------------------------------------------------------------===//

func.func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.SLessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SLessThanEqual
//===----------------------------------------------------------------------===//

func.func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UGreaterThan
//===----------------------------------------------------------------------===//

func.func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.UGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

func.func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ULessThan
//===----------------------------------------------------------------------===//

func.func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.ULessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ULessThanEqual
//===----------------------------------------------------------------------===//

func.func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: spirv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spirv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}
