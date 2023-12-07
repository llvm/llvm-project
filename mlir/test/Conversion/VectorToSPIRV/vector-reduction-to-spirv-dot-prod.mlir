// RUN: mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --test-vector-reduction-to-spirv-dot-prod %s -o - | FileCheck %s

// Positive tests.

// CHECK-LABEL: func.func @to_sdot
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SDot [[ARG0]], [[ARG1]] : (vector<4xi8>, vector<4xi8>) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_sdot(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_sdot_acc
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>, [[ACC:%.+]]: i32)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SDotAccSat [[ARG0]], [[ARG1]], [[ACC]] : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_sdot_acc(%arg0: vector<4xi8>, %arg1: vector<4xi8>, %acc: i32) -> i32 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_sdot_i64
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SDot [[ARG0]], [[ARG1]] : (vector<4xi8>, vector<4xi8>) -> i64
//  CHECK-NEXT:   return [[DOT]] : i64
func.func @to_sdot_i64(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i64 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi64>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi64>
  %mul = arith.muli %lhs, %rhs : vector<4xi64>
  %red = vector.reduction <add>, %mul : vector<4xi64> into i64
  return %red : i64
}

// CHECK-LABEL: func.func @to_sdot_acc_i64
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>, [[ACC:%.+]]: i64)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SDotAccSat [[ARG0]], [[ARG1]], [[ACC]] : (vector<4xi8>, vector<4xi8>, i64) -> i64
//  CHECK-NEXT:   return [[DOT]] : i64
func.func @to_sdot_acc_i64(%arg0: vector<4xi8>, %arg1: vector<4xi8>, %acc: i64) -> i64 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi64>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi64>
  %mul = arith.muli %lhs, %rhs : vector<4xi64>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi64> into i64
  return %red : i64
}

// CHECK-LABEL: func.func @to_udot
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.UDot [[ARG0]], [[ARG1]] : (vector<4xi8>, vector<4xi8>) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_udot(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i32 {
  %lhs = arith.extui %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extui %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_udot_acc
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>, [[ACC:%.+]]: i32)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.UDotAccSat [[ARG0]], [[ARG1]], [[ACC]] : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_udot_acc(%arg0: vector<4xi8>, %arg1: vector<4xi8>, %acc: i32) -> i32 {
  %lhs = arith.extui %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extui %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_signed_unsigned_dot
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SUDot [[ARG0]], [[ARG1]] : (vector<4xi8>, vector<4xi8>) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_signed_unsigned_dot(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extui %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_signed_unsigned_dot_acc
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>, [[ACC:%.+]]: i32)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SUDotAccSat [[ARG0]], [[ARG1]], [[ACC]] : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_signed_unsigned_dot_acc(%arg0: vector<4xi8>, %arg1: vector<4xi8>, %acc: i32) -> i32 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extui %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_unsigned_signed_dot
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SUDot [[ARG1]], [[ARG0]] : (vector<4xi8>, vector<4xi8>) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_unsigned_signed_dot(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i32 {
  %lhs = arith.extui %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_unsigned_signed_dot_acc
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>, [[ACC:%.+]]: i32)
//  CHECK-NEXT:   [[DOT:%.+]] = spirv.SUDotAccSat [[ARG1]], [[ARG0]], [[ACC]] : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-NEXT:   return [[DOT]] : i32
func.func @to_unsigned_signed_dot_acc(%arg0: vector<4xi8>, %arg1: vector<4xi8>, %acc: i32) -> i32 {
  %lhs = arith.extui %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @to_sdot_vector3
//  CHECK-SAME: (%[[ARG0:.+]]: vector<3xi8>, %[[ARG1:.+]]: vector<3xi8>)
//       CHECK:   %[[ZERO:.+]] = spirv.Constant 0 : i8
//       CHECK:   %[[LHS:.+]] = spirv.CompositeConstruct %[[ARG0]], %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//       CHECK:   %[[RHS:.+]] = spirv.CompositeConstruct %[[ARG1]], %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//       CHECK:   %[[SDOT:.+]] = spirv.SDot %[[LHS]], %[[RHS]] : (vector<4xi8>, vector<4xi8>) -> i32
//       CHECK:   return %[[SDOT]]
func.func @to_sdot_vector3(%arg0: vector<3xi8>, %arg1: vector<3xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<3xi8> to vector<3xi32>
  %rhs = arith.extsi %arg1 : vector<3xi8> to vector<3xi32>
  %mul = arith.muli %lhs, %rhs : vector<3xi32>
  %red = vector.reduction <add>, %mul : vector<3xi32> into i32
  return %red : i32
}

// -----

// Negative tests.

// CHECK-LABEL: func.func @too_short
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<2xi8>, [[ARG1:%.+]]: vector<2xi8>)
//  CHECK:        [[RED:%.+]] = vector.reduction
//  CHECK-NEXT:   return [[RED]] : i32
func.func @too_short(%arg0: vector<2xi8>, %arg1: vector<2xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<2xi8> to vector<2xi32>
  %rhs = arith.extsi %arg1 : vector<2xi8> to vector<2xi32>
  %mul = arith.muli %lhs, %rhs : vector<2xi32>
  %red = vector.reduction <add>, %mul : vector<2xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @too_long
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<6xi8>, [[ARG1:%.+]]: vector<6xi8>)
//  CHECK:        [[RED:%.+]] = vector.reduction
//  CHECK-NEXT:   return [[RED]] : i32
func.func @too_long(%arg0: vector<6xi8>, %arg1: vector<6xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<6xi8> to vector<6xi32>
  %rhs = arith.extsi %arg1 : vector<6xi8> to vector<6xi32>
  %mul = arith.muli %lhs, %rhs : vector<6xi32>
  %red = vector.reduction <add>, %mul : vector<6xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @wrong_reduction_kind
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK:        [[RED:%.+]] = vector.reduction <mul>
//  CHECK-NEXT:   return [[RED]] : i32
func.func @wrong_reduction_kind(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <mul>, %mul : vector<4xi32> into i32
  return %red : i32
}

// CHECK-LABEL: func.func @wrong_arith_op
//  CHECK-SAME:   ([[ARG0:%.+]]: vector<4xi8>, [[ARG1:%.+]]: vector<4xi8>)
//  CHECK:        [[ADD:%.+]] = arith.addi
//  CHECK:        [[RED:%.+]] = vector.reduction <mul>, [[ADD]]
//  CHECK-NEXT:   return [[RED]] : i32
func.func @wrong_arith_op(%arg0: vector<4xi8>, %arg1: vector<4xi8>) -> i32 {
  %lhs = arith.extsi %arg0 : vector<4xi8> to vector<4xi32>
  %rhs = arith.extsi %arg1 : vector<4xi8> to vector<4xi32>
  %add = arith.addi %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <mul>, %add : vector<4xi32> into i32
  return %red : i32
}
