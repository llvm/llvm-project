// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @i_equal_scalar
spv.func @i_equal_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : i32
  %0 = spv.IEqual %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @i_equal_vector
spv.func @i_equal_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) "None" {
  // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spv.IEqual %arg0, %arg1 : vector<4xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.INotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @i_not_equal_scalar
spv.func @i_not_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : i64
  %0 = spv.INotEqual %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @i_not_equal_vector
spv.func @i_not_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ne" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.INotEqual %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_greater_than_equal_scalar
spv.func @s_greater_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "sge" %{{.*}}, %{{.*}} : i64
  %0 = spv.SGreaterThanEqual %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @s_greater_than_equal_vector
spv.func @s_greater_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "sge" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_greater_than_scalar
spv.func @s_greater_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "sgt" %{{.*}}, %{{.*}} : i64
  %0 = spv.SGreaterThan %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @s_greater_than_vector
spv.func @s_greater_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "sgt" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.SGreaterThan %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SLessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_less_than_equal_scalar
spv.func @s_less_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "sle" %{{.*}}, %{{.*}} : i64
  %0 = spv.SLessThanEqual %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @s_less_than_equal_vector
spv.func @s_less_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "sle" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.SLessThanEqual %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SLessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @s_less_than_scalar
spv.func @s_less_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
  %0 = spv.SLessThan %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @s_less_than_vector
spv.func @s_less_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "slt" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.SLessThan %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_greater_than_equal_scalar
spv.func @u_greater_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "uge" %{{.*}}, %{{.*}} : i64
  %0 = spv.UGreaterThanEqual %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @u_greater_than_equal_vector
spv.func @u_greater_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "uge" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.UGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_greater_than_scalar
spv.func @u_greater_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ugt" %{{.*}}, %{{.*}} : i64
  %0 = spv.UGreaterThan %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @u_greater_than_vector
spv.func @u_greater_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ugt" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.UGreaterThan %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ULessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_less_than_equal_scalar
spv.func @u_less_than_equal_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ule" %{{.*}}, %{{.*}} : i64
  %0 = spv.ULessThanEqual %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @u_less_than_equal_vector
spv.func @u_less_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ule" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.ULessThanEqual %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ULessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @u_less_than_scalar
spv.func @u_less_than_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.icmp "ult" %{{.*}}, %{{.*}} : i64
  %0 = spv.ULessThan %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @u_less_than_vector
spv.func @u_less_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.icmp "ult" %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.ULessThan %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FOrdEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_equal_scalar
spv.func @f_ord_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "oeq" %{{.*}}, %{{.*}} : f32
  %0 = spv.FOrdEqual %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @f_ord_equal_vector
spv.func @f_ord_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "oeq" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spv.FOrdEqual %arg0, %arg1 : vector<4xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FOrdGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_greater_than_equal_scalar
spv.func @f_ord_greater_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "oge" %{{.*}}, %{{.*}} : f64
  %0 = spv.FOrdGreaterThanEqual %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_ord_greater_than_equal_vector
spv.func @f_ord_greater_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "oge" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FOrdGreaterThanEqual %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FOrdGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_greater_than_scalar
spv.func @f_ord_greater_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ogt" %{{.*}}, %{{.*}} : f64
  %0 = spv.FOrdGreaterThan %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_ord_greater_than_vector
spv.func @f_ord_greater_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ogt" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FOrdGreaterThan %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FOrdLessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_less_than_scalar
spv.func @f_ord_less_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "olt" %{{.*}}, %{{.*}} : f64
  %0 = spv.FOrdLessThan %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_ord_less_than_vector
spv.func @f_ord_less_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "olt" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FOrdLessThan %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FOrdLessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_less_than_equal_scalar
spv.func @f_ord_less_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ole" %{{.*}}, %{{.*}} : f64
  %0 = spv.FOrdLessThanEqual %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_ord_less_than_equal_vector
spv.func @f_ord_less_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ole" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FOrdLessThanEqual %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FOrdNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_ord_not_equal_scalar
spv.func @f_ord_not_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "one" %{{.*}}, %{{.*}} : f32
  %0 = spv.FOrdNotEqual %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @f_ord_not_equal_vector
spv.func @f_ord_not_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "one" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spv.FOrdNotEqual %arg0, %arg1 : vector<4xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FUnordEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_equal_scalar
spv.func @f_unord_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "ueq" %{{.*}}, %{{.*}} : f32
  %0 = spv.FUnordEqual %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @f_unord_equal_vector
spv.func @f_unord_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "ueq" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spv.FUnordEqual %arg0, %arg1 : vector<4xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FUnordGreaterThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_greater_than_equal_scalar
spv.func @f_unord_greater_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "uge" %{{.*}}, %{{.*}} : f64
  %0 = spv.FUnordGreaterThanEqual %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_unord_greater_than_equal_vector
spv.func @f_unord_greater_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "uge" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FUnordGreaterThanEqual %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FUnordGreaterThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_greater_than_scalar
spv.func @f_unord_greater_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ugt" %{{.*}}, %{{.*}} : f64
  %0 = spv.FUnordGreaterThan %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_unord_greater_than_vector
spv.func @f_unord_greater_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ugt" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FUnordGreaterThan %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FUnordLessThan
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_less_than_scalar
spv.func @f_unord_less_than_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ult" %{{.*}}, %{{.*}} : f64
  %0 = spv.FUnordLessThan %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_unord_less_than_vector
spv.func @f_unord_less_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ult" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FUnordLessThan %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FUnordLessThanEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_less_than_equal_scalar
spv.func @f_unord_less_than_equal_scalar(%arg0: f64, %arg1: f64) "None" {
  // CHECK: llvm.fcmp "ule" %{{.*}}, %{{.*}} : f64
  %0 = spv.FUnordLessThanEqual %arg0, %arg1 : f64
  spv.Return
}

// CHECK-LABEL: @f_unord_less_than_equal_vector
spv.func @f_unord_less_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fcmp "ule" %{{.*}}, %{{.*}} : vector<2xf64>
  %0 = spv.FUnordLessThanEqual %arg0, %arg1 : vector<2xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FUnordNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @f_unord_not_equal_scalar
spv.func @f_unord_not_equal_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fcmp "une" %{{.*}}, %{{.*}} : f32
  %0 = spv.FUnordNotEqual %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @f_unord_not_equal_vector
spv.func @f_unord_not_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) "None" {
  // CHECK: llvm.fcmp "une" %{{.*}}, %{{.*}} : vector<4xf64>
  %0 = spv.FUnordNotEqual %arg0, %arg1 : vector<4xf64>
  spv.Return
}
