// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.IEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @i_equal_scalar
spirv.func @i_equal_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : i32
  %0 = spirv.IEqual %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @i_equal_vector
spirv.func @i_equal_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spirv.IEqual %arg0, %arg1 : vector<4xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.INotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @i_not_equal_scalar
spirv.func @i_not_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : i64
  %0 = spirv.INotEqual %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @i_not_equal_vector
spirv.func @i_not_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.INotEqual %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_greater_than_equal_scalar
spirv.func @s_greater_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "sge" %{{.*}}, %{{.*}} : i64
  %0 = spirv.SGreaterThanEqual %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @s_greater_than_equal_vector
spirv.func @s_greater_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "sge" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.SGreaterThanEqual %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_greater_than_scalar
spirv.func @s_greater_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "sgt" %{{.*}}, %{{.*}} : i64
  %0 = spirv.SGreaterThan %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @s_greater_than_vector
spirv.func @s_greater_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "sgt" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.SGreaterThan %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SLessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_less_than_equal_scalar
spirv.func @s_less_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "sle" %{{.*}}, %{{.*}} : i64
  %0 = spirv.SLessThanEqual %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @s_less_than_equal_vector
spirv.func @s_less_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "sle" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.SLessThanEqual %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SLessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_less_than_scalar
spirv.func @s_less_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
  %0 = spirv.SLessThan %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @s_less_than_vector
spirv.func @s_less_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.SLessThan %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_greater_than_equal_scalar
spirv.func @u_greater_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "uge" %{{.*}}, %{{.*}} : i64
  %0 = spirv.UGreaterThanEqual %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @u_greater_than_equal_vector
spirv.func @u_greater_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "uge" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.UGreaterThanEqual %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.UGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_greater_than_scalar
spirv.func @u_greater_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ugt" %{{.*}}, %{{.*}} : i64
  %0 = spirv.UGreaterThan %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @u_greater_than_vector
spirv.func @u_greater_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ugt" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.UGreaterThan %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ULessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_less_than_equal_scalar
spirv.func @u_less_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ule" %{{.*}}, %{{.*}} : i64
  %0 = spirv.ULessThanEqual %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @u_less_than_equal_vector
spirv.func @u_less_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ule" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.ULessThanEqual %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ULessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_less_than_scalar
spirv.func @u_less_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ult" %{{.*}}, %{{.*}} : i64
  %0 = spirv.ULessThan %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @u_less_than_vector
spirv.func @u_less_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ult" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.ULessThan %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FOrdEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_equal_scalar
spirv.func @f_ord_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "oeq" %{{.*}}, %{{.*}} : f32
  %0 = spirv.FOrdEqual %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @f_ord_equal_vector
spirv.func @f_ord_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "oeq" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spirv.FOrdEqual %arg0, %arg1 : vector<4xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FOrdGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_greater_than_equal_scalar
spirv.func @f_ord_greater_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "oge" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FOrdGreaterThanEqual %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_ord_greater_than_equal_vector
spirv.func @f_ord_greater_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "oge" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FOrdGreaterThanEqual %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FOrdGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_greater_than_scalar
spirv.func @f_ord_greater_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ogt" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FOrdGreaterThan %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_ord_greater_than_vector
spirv.func @f_ord_greater_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ogt" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FOrdGreaterThan %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FOrdLessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_less_than_scalar
spirv.func @f_ord_less_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "olt" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FOrdLessThan %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_ord_less_than_vector
spirv.func @f_ord_less_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "olt" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FOrdLessThan %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FOrdLessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_less_than_equal_scalar
spirv.func @f_ord_less_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ole" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FOrdLessThanEqual %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_ord_less_than_equal_vector
spirv.func @f_ord_less_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ole" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FOrdLessThanEqual %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FOrdNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_not_equal_scalar
spirv.func @f_ord_not_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "one" %{{.*}}, %{{.*}} : f32
  %0 = spirv.FOrdNotEqual %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @f_ord_not_equal_vector
spirv.func @f_ord_not_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "one" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spirv.FOrdNotEqual %arg0, %arg1 : vector<4xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FUnordEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_equal_scalar
spirv.func @f_unord_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "ueq" %{{.*}}, %{{.*}} : f32
  %0 = spirv.FUnordEqual %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @f_unord_equal_vector
spirv.func @f_unord_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "ueq" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spirv.FUnordEqual %arg0, %arg1 : vector<4xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FUnordGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_greater_than_equal_scalar
spirv.func @f_unord_greater_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "uge" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FUnordGreaterThanEqual %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_unord_greater_than_equal_vector
spirv.func @f_unord_greater_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "uge" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FUnordGreaterThanEqual %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FUnordGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_greater_than_scalar
spirv.func @f_unord_greater_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ugt" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FUnordGreaterThan %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_unord_greater_than_vector
spirv.func @f_unord_greater_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ugt" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FUnordGreaterThan %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FUnordLessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_less_than_scalar
spirv.func @f_unord_less_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ult" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FUnordLessThan %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_unord_less_than_vector
spirv.func @f_unord_less_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ult" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FUnordLessThan %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FUnordLessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_less_than_equal_scalar
spirv.func @f_unord_less_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ule" %{{.*}}, %{{.*}} : f64
  %0 = spirv.FUnordLessThanEqual %arg0, %arg1 : f64
  spirv.Return
}

// CHECK-LABEL: @f_unord_less_than_equal_vector
spirv.func @f_unord_less_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ule" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spirv.FUnordLessThanEqual %arg0, %arg1 : vector<2xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FUnordNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_not_equal_scalar
spirv.func @f_unord_not_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "une" %{{.*}}, %{{.*}} : f32
  %0 = spirv.FUnordNotEqual %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @f_unord_not_equal_vector
spirv.func @f_unord_not_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "une" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spirv.FUnordNotEqual %arg0, %arg1 : vector<4xf64>
  spirv.Return
}
