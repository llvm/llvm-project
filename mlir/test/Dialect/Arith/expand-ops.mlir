// RUN: mlir-opt %s -arith-expand="include-bf16=true include-f8e8m0=true include-f4e2m1=true" -verify-diagnostics -split-input-file | FileCheck %s
// RUN: mlir-opt %s -arith-expand -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=SCHECK
// RUN: mlir-opt %s -arith-expand="include-bf16=true include-f8e8m0=true include-f4e2m1=true" -canonicalize -verify-diagnostics -split-input-file | FileCheck %s --check-prefix=VALUES

// CHECK-LABEL: func.func @ceildivui(
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32) -> i32 {
// CHECK-NOT: arith.ceildivui
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divui %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : i32
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : i32
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[INEXACT]] : i1 to i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32
func.func @ceildivui(%arg0: i32, %arg1: i32) -> i32 {
  %res = arith.ceildivui %arg0, %arg1 : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @ceildivui_index(
// CHECK-SAME: %[[LHS:.*]]: index, %[[RHS:.*]]: index) -> index {
// CHECK-NOT: arith.ceildivui
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.extui
// CHECK-NEXT: %[[Q:.*]] = arith.divui %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : index
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : index
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.index_castui %[[INEXACT]] : i1 to index
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : index
// CHECK-NEXT: return %[[RESULT]] : index
func.func @ceildivui_index(%arg0: index, %arg1: index) -> index {
  %res = arith.ceildivui %arg0, %arg1 : index
  return %res : index
}

// -----

// CHECK-LABEL: func.func @ceildivui_vec(
// CHECK-SAME: %[[LHS:.*]]: vector<4xi32>, %[[RHS:.*]]: vector<4xi32>) -> vector<4xi32> {
// CHECK-NOT: arith.ceildivui
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divui %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : vector<4xi32>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[INEXACT]] : vector<4xi1> to vector<4xi32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : vector<4xi32>
// CHECK-NEXT: return %[[RESULT]] : vector<4xi32>
func.func @ceildivui_vec(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
  %res = arith.ceildivui %arg0, %arg1 : vector<4xi32>
  return %res : vector<4xi32>
}

// -----

// CHECK-LABEL: func.func @ceildivui_static_tensor(
// CHECK-SAME: %[[LHS:.*]]: tensor<2x3xi32>, %[[RHS:.*]]: tensor<2x3xi32>) -> tensor<2x3xi32> {
// CHECK-NOT: arith.ceildivui
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divui %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[INEXACT]] : tensor<2x3xi1> to tensor<2x3xi32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : tensor<2x3xi32>
// CHECK-NEXT: return %[[RESULT]] : tensor<2x3xi32>
func.func @ceildivui_static_tensor(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %res = arith.ceildivui %arg0, %arg1 : tensor<2x3xi32>
  return %res : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func.func @ceildivui_dynamic_tensor(
// CHECK-SAME: %[[LHS:.*]]: tensor<8x4x?xi64>, %[[RHS:.*]]: tensor<8x4x?xi64>) -> tensor<8x4x?xi64> {
// CHECK-NOT: arith.ceildivui
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divui %[[LHS]], %[[RHS]] : tensor<8x4x?xi64>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : tensor<8x4x?xi64>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : tensor<8x4x?xi64>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[INEXACT]] : tensor<8x4x?xi1> to tensor<8x4x?xi64>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : tensor<8x4x?xi64>
// CHECK-NEXT: return %[[RESULT]] : tensor<8x4x?xi64>
func.func @ceildivui_dynamic_tensor(%arg0: tensor<8x4x?xi64>, %arg1: tensor<8x4x?xi64>) -> tensor<8x4x?xi64> {
  %res = arith.ceildivui %arg0, %arg1 : tensor<8x4x?xi64>
  return %res : tensor<8x4x?xi64>
}

// -----

// CHECK-LABEL: func.func @ceildivui_i1(
// CHECK-SAME: %[[LHS:.*]]: i1, %[[RHS:.*]]: i1) -> i1 {
// CHECK-NOT: arith.ceildivui
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divui %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : i1
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : i1
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[INEXACT]] : i1
// CHECK-NEXT: return %[[RESULT]] : i1
func.func @ceildivui_i1(%arg0: i1, %arg1: i1) -> i1 {
  %res = arith.ceildivui %arg0, %arg1 : i1
  return %res : i1
}

// -----

// CHECK-LABEL: func.func @ceildivi(
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32) -> i32 {
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : i32
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : i32
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[SAME_SIGN:.*]] = arith.cmpi eq, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[SAME_SIGN]] : i1
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : i1 to i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32
func.func @ceildivi(%arg0: i32, %arg1: i32) -> i32 {
  %res = arith.ceildivsi %arg0, %arg1 : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @ceildivi_index(
// CHECK-SAME: %[[LHS:.*]]: index, %[[RHS:.*]]: index) -> index {
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.extui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : index
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : index
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[SAME_SIGN:.*]] = arith.cmpi eq, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[SAME_SIGN]] : i1
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.index_castui %[[ROUND]] : i1 to index
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : index
// CHECK-NEXT: return %[[RESULT]] : index
func.func @ceildivi_index(%arg0: index, %arg1: index) -> index {
  %res = arith.ceildivsi %arg0, %arg1 : index
  return %res : index
}

// -----

// CHECK-LABEL: func.func @ceildivsi_vec(
// CHECK-SAME: %[[LHS:.*]]: vector<4xi32>, %[[RHS:.*]]: vector<4xi32>) -> vector<4xi32> {
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : vector<4xi32>
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[SAME_SIGN:.*]] = arith.cmpi eq, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : vector<4xi1>
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[SAME_SIGN]] : vector<4xi1>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : vector<4xi1> to vector<4xi32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : vector<4xi32>
// CHECK-NEXT: return %[[RESULT]] : vector<4xi32>
func.func @ceildivsi_vec(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
  %res = arith.ceildivsi %arg0, %arg1 : vector<4xi32>
  return %res : vector<4xi32>
}

// -----

// CHECK-LABEL: func.func @ceildivsi_static_tensor(
// CHECK-SAME: %[[LHS:.*]]: tensor<2x3xi32>, %[[RHS:.*]]: tensor<2x3xi32>) -> tensor<2x3xi32> {
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[SAME_SIGN:.*]] = arith.cmpi eq, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : tensor<2x3xi1>
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[SAME_SIGN]] : tensor<2x3xi1>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : tensor<2x3xi1> to tensor<2x3xi32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : tensor<2x3xi32>
// CHECK-NEXT: return %[[RESULT]] : tensor<2x3xi32>
func.func @ceildivsi_static_tensor(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %res = arith.ceildivsi %arg0, %arg1 : tensor<2x3xi32>
  return %res : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func.func @ceildivsi_dynamic_tensor(
// CHECK-SAME: %[[LHS:.*]]: tensor<8x?xi64>, %[[RHS:.*]]: tensor<8x?xi64>) -> tensor<8x?xi64> {
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : tensor<8x?xi64>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : tensor<8x?xi64>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : tensor<8x?xi64>
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : tensor<8x?xi64>
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : tensor<8x?xi64>
// CHECK-NEXT: %[[SAME_SIGN:.*]] = arith.cmpi eq, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : tensor<8x?xi1>
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[SAME_SIGN]] : tensor<8x?xi1>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : tensor<8x?xi1> to tensor<8x?xi64>
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ADJUSTMENT]] : tensor<8x?xi64>
// CHECK-NEXT: return %[[RESULT]] : tensor<8x?xi64>
func.func @ceildivsi_dynamic_tensor(%arg0: tensor<8x?xi64>, %arg1: tensor<8x?xi64>) -> tensor<8x?xi64> {
  %res = arith.ceildivsi %arg0, %arg1 : tensor<8x?xi64>
  return %res : tensor<8x?xi64>
}

// -----

// CHECK-LABEL: func.func @ceildivsi_i1(
// CHECK-SAME: %[[LHS:.*]]: i1, %[[RHS:.*]]: i1) -> i1 {
// CHECK-NOT: arith.ceildivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : i1
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : i1
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[SAME_SIGN:.*]] = arith.cmpi eq, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[SAME_SIGN]] : i1
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[Q]], %[[ROUND]] : i1
// CHECK-NEXT: return %[[RESULT]] : i1
func.func @ceildivsi_i1(%arg0: i1, %arg1: i1) -> i1 {
  %res = arith.ceildivsi %arg0, %arg1 : i1
  return %res : i1
}

// -----

// CHECK-LABEL: func.func @floordivi(
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32) -> i32 {
// CHECK-NOT: arith.floordivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : i32
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : i32
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[OPPOSITE_SIGN]] : i1
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : i1 to i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.subi %[[Q]], %[[ADJUSTMENT]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32
func.func @floordivi(%arg0: i32, %arg1: i32) -> i32 {
  %res = arith.floordivsi %arg0, %arg1 : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @floordivi_index(
// CHECK-SAME: %[[LHS:.*]]: index, %[[RHS:.*]]: index) -> index {
// CHECK-NOT: arith.floordivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.extui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : index
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : index
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : index
// CHECK-NEXT: %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[OPPOSITE_SIGN]] : i1
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.index_castui %[[ROUND]] : i1 to index
// CHECK-NEXT: %[[RESULT:.*]] = arith.subi %[[Q]], %[[ADJUSTMENT]] : index
// CHECK-NEXT: return %[[RESULT]] : index
func.func @floordivi_index(%arg0: index, %arg1: index) -> index {
  %res = arith.floordivsi %arg0, %arg1 : index
  return %res : index
}

// -----

// CHECK-LABEL: func.func @floordivi_vec(
// CHECK-SAME: %[[LHS:.*]]: vector<4xi32>, %[[RHS:.*]]: vector<4xi32>) -> vector<4xi32> {
// CHECK-NOT: arith.floordivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : vector<4xi32>
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK-NEXT: %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : vector<4xi1>
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[OPPOSITE_SIGN]] : vector<4xi1>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : vector<4xi1> to vector<4xi32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.subi %[[Q]], %[[ADJUSTMENT]] : vector<4xi32>
// CHECK-NEXT: return %[[RESULT]] : vector<4xi32>
func.func @floordivi_vec(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
  %res = arith.floordivsi %arg0, %arg1 : vector<4xi32>
  return %res : vector<4xi32>
}

// -----

// CHECK-LABEL: func.func @floordivsi_static_tensor(
// CHECK-SAME: %[[LHS:.*]]: tensor<2x3xi32>, %[[RHS:.*]]: tensor<2x3xi32>) -> tensor<2x3xi32> {
// CHECK-NOT: arith.floordivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : tensor<2x3xi32>
// CHECK-NEXT: %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : tensor<2x3xi1>
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[OPPOSITE_SIGN]] : tensor<2x3xi1>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : tensor<2x3xi1> to tensor<2x3xi32>
// CHECK-NEXT: %[[RESULT:.*]] = arith.subi %[[Q]], %[[ADJUSTMENT]] : tensor<2x3xi32>
// CHECK-NEXT: return %[[RESULT]] : tensor<2x3xi32>
func.func @floordivsi_static_tensor(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %res = arith.floordivsi %arg0, %arg1 : tensor<2x3xi32>
  return %res : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func.func @floordivsi_dynamic_tensor(
// CHECK-SAME: %[[LHS:.*]]: tensor<?x4xi64>, %[[RHS:.*]]: tensor<?x4xi64>) -> tensor<?x4xi64> {
// CHECK-NOT: arith.floordivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : tensor<?x4xi64>
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : tensor<?x4xi64>
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : tensor<?x4xi64>
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : tensor<?x4xi64>
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : tensor<?x4xi64>
// CHECK-NEXT: %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : tensor<?x4xi1>
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[OPPOSITE_SIGN]] : tensor<?x4xi1>
// CHECK-NEXT: %[[ADJUSTMENT:.*]] = arith.extui %[[ROUND]] : tensor<?x4xi1> to tensor<?x4xi64>
// CHECK-NEXT: %[[RESULT:.*]] = arith.subi %[[Q]], %[[ADJUSTMENT]] : tensor<?x4xi64>
// CHECK-NEXT: return %[[RESULT]] : tensor<?x4xi64>
func.func @floordivsi_dynamic_tensor(%arg0: tensor<?x4xi64>, %arg1: tensor<?x4xi64>) -> tensor<?x4xi64> {
  %res = arith.floordivsi %arg0, %arg1 : tensor<?x4xi64>
  return %res : tensor<?x4xi64>
}

// -----

// CHECK-LABEL: func.func @floordivsi_i1(
// CHECK-SAME: %[[LHS:.*]]: i1, %[[RHS:.*]]: i1) -> i1 {
// CHECK-NOT: arith.floordivsi
// CHECK-NOT: arith.constant
// CHECK-NOT: tensor.
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.index_castui
// CHECK-NEXT: %[[Q:.*]] = arith.divsi %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.muli %[[Q]], %[[RHS]] : i1
// CHECK-NEXT: %[[INEXACT:.*]] = arith.cmpi ne, %[[PRODUCT]], %[[LHS]] : i1
// CHECK-NEXT: %[[SIGNED_LT:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[UNSIGNED_LT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i1
// CHECK-NEXT: %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
// CHECK-NEXT: %[[ROUND:.*]] = arith.andi %[[INEXACT]], %[[OPPOSITE_SIGN]] : i1
// CHECK-NEXT: %[[RESULT:.*]] = arith.subi %[[Q]], %[[ROUND]] : i1
// CHECK-NEXT: return %[[RESULT]] : i1
func.func @floordivsi_i1(%arg0: i1, %arg1: i1) -> i1 {
  %res = arith.floordivsi %arg0, %arg1 : i1
  return %res : i1
}

// -----

// VALUES-LABEL: func.func @ceildivui_values()
// VALUES-DAG: %[[THREE:.*]] = arith.constant 3 : i8
// VALUES-DAG: %[[TWO:.*]] = arith.constant 2 : i8
// VALUES-DAG: %[[ZERO:.*]] = arith.constant 0 : i8
// VALUES: return %[[THREE]], %[[TWO]], %[[ZERO]] : i8, i8, i8
func.func @ceildivui_values() -> (i8, i8, i8) {
  %zero = arith.constant 0 : i8
  %two = arith.constant 2 : i8
  %four = arith.constant 4 : i8
  %five = arith.constant 5 : i8
  %inexact = arith.ceildivui %five, %two : i8
  %exact = arith.ceildivui %four, %two : i8
  %zeroDividend = arith.ceildivui %zero, %two : i8
  return %inexact, %exact, %zeroDividend : i8, i8, i8
}

// VALUES-LABEL: func.func @ceildivsi_values()
// VALUES-DAG: %[[THREE:.*]] = arith.constant 3 : i8
// VALUES-DAG: %[[NEG_TWO:.*]] = arith.constant -2 : i8
// VALUES: return %[[THREE]], %[[NEG_TWO]], %[[NEG_TWO]], %[[THREE]] : i8, i8, i8, i8
func.func @ceildivsi_values() -> (i8, i8, i8, i8) {
  %negFive = arith.constant -5 : i8
  %negTwo = arith.constant -2 : i8
  %two = arith.constant 2 : i8
  %five = arith.constant 5 : i8
  %posPos = arith.ceildivsi %five, %two : i8
  %negPos = arith.ceildivsi %negFive, %two : i8
  %posNeg = arith.ceildivsi %five, %negTwo : i8
  %negNeg = arith.ceildivsi %negFive, %negTwo : i8
  return %posPos, %negPos, %posNeg, %negNeg : i8, i8, i8, i8
}

// VALUES-LABEL: func.func @floordivsi_values()
// VALUES-DAG: %[[TWO:.*]] = arith.constant 2 : i8
// VALUES-DAG: %[[NEG_THREE:.*]] = arith.constant -3 : i8
// VALUES: return %[[TWO]], %[[NEG_THREE]], %[[NEG_THREE]], %[[TWO]] : i8, i8, i8, i8
func.func @floordivsi_values() -> (i8, i8, i8, i8) {
  %negFive = arith.constant -5 : i8
  %negTwo = arith.constant -2 : i8
  %two = arith.constant 2 : i8
  %five = arith.constant 5 : i8
  %posPos = arith.floordivsi %five, %two : i8
  %negPos = arith.floordivsi %negFive, %two : i8
  %posNeg = arith.floordivsi %five, %negTwo : i8
  %negNeg = arith.floordivsi %negFive, %negTwo : i8
  return %posPos, %negPos, %posNeg, %negNeg : i8, i8, i8, i8
}

// -----

// CHECK-LABEL: func @maximumf
func.func @maximumf(%a: f32, %b: f32) -> f32 {
  %result = arith.maximumf %a, %b : f32
  return %result : f32
}
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ugt, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maximumf_vector
func.func @maximumf_vector(%a: vector<4xf16>, %b: vector<4xf16>) -> vector<4xf16> {
  %result = arith.maximumf %a, %b : vector<4xf16>
  return %result : vector<4xf16>
}
// CHECK-SAME: %[[LHS:.*]]: vector<4xf16>, %[[RHS:.*]]: vector<4xf16>)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ugt, %[[LHS]], %[[RHS]] : vector<4xf16>
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]]
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RHS]], %[[RHS]] : vector<4xf16>
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]]
// CHECK-NEXT: return %[[RESULT]] : vector<4xf16>

// -----

// CHECK-LABEL: func @maxnumf
func.func @maxnumf(%a: f32, %b: f32) -> f32 {
  %result = arith.maxnumf %a, %b : f32
  return %result : f32
}

// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ugt, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[LHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @minimumf
func.func @minimumf(%a: f32, %b: f32) -> f32 {
  %result = arith.minimumf %a, %b : f32
  return %result : f32
}

// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ult, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @minnumf
func.func @minnumf(%a: f32, %b: f32) -> f32 {
  %result = arith.minnumf %a, %b : f32
  return %result : f32
}

// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ult, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[LHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

func.func @truncf_f32(%arg0 : f32) -> bf16 {
    %0 = arith.truncf %arg0 : f32 to bf16
    return %0 : bf16
}

// CHECK-LABEL: @truncf_f32
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[C7FC0_i16:.+]] = arith.constant 32704 : i16
// CHECK-DAG: %[[C7FFF:.+]] = arith.constant 32767 : i32
// CHECK-DAG: %[[ISNAN:.+]] = arith.cmpf une, %arg0, %arg0 : f32
// CHECK-DAG: %[[BITCAST:.+]] = arith.bitcast %arg0 : f32 to i32
// CHECK-DAG: %[[SHRUI:.+]] = arith.shrui %[[BITCAST]], %[[C16]] : i32
// CHECK-DAG: %[[BIT16:.+]] = arith.andi %[[SHRUI]], %[[C1]] : i32
// CHECK-DAG: %[[ROUNDING_BIAS:.+]] = arith.addi %[[BIT16]], %[[C7FFF]] : i32
// CHECK-DAG: %[[BIASED:.+]] = arith.addi %[[BITCAST]], %[[ROUNDING_BIAS]] : i32
// CHECK-DAG: %[[BIASED_SHIFTED:.+]] = arith.shrui %[[BIASED]], %[[C16]] : i32
// CHECK-DAG: %[[NORMAL_CASE_RESULT_i16:.+]] = arith.trunci %[[BIASED_SHIFTED]] : i32 to i16
// CHECK-DAG: %[[SELECT:.+]] = arith.select %[[ISNAN]], %[[C7FC0_i16]], %[[NORMAL_CASE_RESULT_i16]] : i16
// CHECK-DAG: %[[RESULT:.+]] = arith.bitcast %[[SELECT]] : i16 to bf16
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_vector_f32(%arg0 : vector<4xf32>) -> vector<4xbf16> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @truncf_vector_f32
// CHECK-NOT: arith.truncf

// -----
func.func @truncf_f32_to_f8E8M0FNU(%arg0 : f32) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 : f32 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f32_to_f8E8M0FNU
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f32 to i32
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[SHRUI:.+]] = arith.shrui %[[BITCAST]], %[[C23_i32]] : i32
// CHECK: %[[TRUNCI:.+]] = arith.trunci %[[SHRUI]] : i32 to i8
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[TRUNCI]] : i8 to f8E8M0FNU
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_f16_to_f8E8M0FNU(%arg0 : f16) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 : f16 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f16_to_f8E8M0FNU
// CHECK: %[[EXTF:.+]] = arith.extf %arg0 : f16 to f32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %[[EXTF]] : f32 to i32
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[SHRUI:.+]] = arith.shrui %[[BITCAST]], %[[C23_i32]] : i32
// CHECK: %[[TRUNCI:.+]] = arith.trunci %[[SHRUI]] : i32 to i8
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[TRUNCI]] : i8 to f8E8M0FNU
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_vector_f32_to_f8E8M0FNU(%arg0 : vector<4xf32>) -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}

// CHECK-LABEL: @truncf_vector_f32_to_f8E8M0FNU
// CHECK-NOT: arith.truncf

// -----

func.func @truncf_vector_f16_to_f8E8M0FNU(%arg0 : vector<4xf16>) -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 : vector<4xf16> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}

// CHECK-LABEL: @truncf_vector_f16_to_f8E8M0FNU
// CHECK-NOT: arith.truncf

// -----

func.func @truncf_vector_bf16_to_f8E8M0FNU(%arg0 : vector<4xbf16>) -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 : vector<4xbf16> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}

// CHECK-LABEL: @truncf_vector_bf16_to_f8E8M0FNU
// CHECK-NOT: arith.truncf
// CHECK: return

// -----

func.func @scaling_truncf_f32_to_f4E2M1FN(%arg0 : f32, %arg1: f8E8M0FNU) -> f4E2M1FN {
    %0 = arith.scaling_truncf %arg0, %arg1 : f32, f8E8M0FNU to f4E2M1FN
    return %0 : f4E2M1FN
}

// SCHECK-LABEL: @scaling_truncf_f32_to_f4E2M1FN
// SCHECK: %[[SCALEF32:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// SCHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEF32]] : f32
// SCHECK: %[[RESULT:.+]] = arith.truncf %[[DIVF]] : f32 to f4E2M1FN
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_truncf_vector_f16_to_f6E3M2FN(%arg0 : vector<4xf16>, %arg1: vector<4xf8E8M0FNU>) -> vector<4xf6E3M2FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 : vector<4xf16>, vector<4xf8E8M0FNU> to vector<4xf6E3M2FN>
    return %0 : vector<4xf6E3M2FN>
}

// SCHECK-LABEL: @scaling_truncf_vector_f16_to_f6E3M2FN
// SCHECK: %[[SCALEF16:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf16>
// SCHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEF16]] : vector<4xf16>
// SCHECK: %[[RESULT:.+]] = arith.truncf %[[DIVF]] : vector<4xf16> to vector<4xf6E3M2FN>
// SCHECK: return %[[RESULT]] : vector<4xf6E3M2FN>

// -----

func.func @scaling_truncf_propagate_rounding_mode_fast_math(%arg0 : vector<4xf16>, %arg1: vector<4xf16>) -> vector<4xf6E3M2FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 to_nearest_even fastmath<fast> : vector<4xf16>, vector<4xf16> to vector<4xf6E3M2FN>
    return %0 : vector<4xf6E3M2FN>
}
// SCHECK-LABEL: @scaling_truncf_propagate_rounding_mode_fast_math
// SCHECK: %[[SCALEF8:.+]] = arith.truncf %arg1 fastmath<fast> : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: %[[SCALEINTY:.+]] = arith.extf %[[SCALEF8]] fastmath<fast> : vector<4xf8E8M0FNU> to vector<4xf16>
// SCHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEINTY]] fastmath<fast> : vector<4xf16>
// SCHECK: %[[TRUNCF:.+]] = arith.truncf [[_:%[a-zA-Z0-9_]+]] to_nearest_even fastmath<fast> : vector<4xf16> to vector<4xf6E3M2FN>
// SCHECK: return %[[TRUNCF]] : vector<4xf6E3M2FN>

// -----

func.func @scaling_truncf_f16_to_f4E2M1FN_using_f16_scales(%arg0: f16, %arg1 : f16) -> f4E2M1FN {
    %0 = arith.scaling_truncf %arg0, %arg1 : f16, f16 to f4E2M1FN
    return %0 : f4E2M1FN
}
// SCHECK-LABEL: @scaling_truncf_f16_to_f4E2M1FN_using_f16_scales
// SCHECK: %[[SCALETRUNCF:.+]] = arith.truncf %arg1 : f16 to f8E8M0FN
// SCHECK: return

// -----
func.func @scaling_truncf_vector_f16_to_f4E2M1FN_using_f16_scales(%arg0: vector<4xf16>, %arg1 : vector<4xf16>) -> vector<4xf4E2M1FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 : vector<4xf16>, vector<4xf16> to vector<4xf4E2M1FN>
    return %0 : vector<4xf4E2M1FN>
}
// SCHECK-LABEL: @scaling_truncf_vector_f16_to_f4E2M1FN_using_f16_scales
// SCHECK: %[[SCALETRUNCF:.+]] = arith.truncf %arg1 : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: return

// -----

func.func @invalid_scaling_truncf_to_f4E2M1FN(%arg0: f16, %arg1 : f8E5M2FNUZ) -> f4E2M1FN {
    // expected-error@+1 {{failed to legalize operation 'arith.scaling_truncf' that was explicitly marked illegal}}
    %0 = arith.scaling_truncf %arg0, %arg1 : f16, f8E5M2FNUZ to f4E2M1FN
    return %0 : f4E2M1FN
}

// -----

func.func @extf_f8E8M0FNU_to_f32(%arg0 : f8E8M0FNU) -> f32 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f8E8M0FNU_to_f32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF32NAN:.+]] = arith.constant -1 : i32
// CHECK: %[[CMP_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT_NAN:.+]] = arith.select %[[CMP_NAN]], %[[CF32NAN]], %[[SHLI]] : i32
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[SELECT_NAN]] : i32 to f32
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f32_no_nan(%arg0 : f8E8M0FNU) -> f32 {
    %0 = arith.extf %arg0 fastmath<nnan> : f8E8M0FNU to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f8E8M0FNU_to_f32_no_nan
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[SHLI]] : i32 to f32
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f16(%arg0 : f8E8M0FNU) -> f16 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f16
    return %0 : f16
}

// CHECK-LABEL: @extf_f8E8M0FNU_to_f16
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK-DAG: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF32NAN:.+]] = arith.constant -1 : i32
// CHECK: %[[CMP_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT_NAN:.+]] = arith.select %[[CMP_NAN]], %[[CF32NAN]], %[[SHLI]] : i32
// CHECK: %[[F32_RESULT:.+]] = arith.bitcast %[[SELECT_NAN]] : i32 to f32
// CHECK: %[[F16_RESULT:.+]] = arith.truncf %[[F32_RESULT]] : f32 to f16
// CHECK: return %[[F16_RESULT]]

// -----

func.func @extf_vector_f8E8M0FNU_to_f32(%arg0 : vector<4xf8E8M0FNU>) -> vector<4xf32> {
    %0 = arith.extf %arg0 : vector<4xf8E8M0FNU> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @extf_vector_f8E8M0FNU_to_f32
// CHECK-NOT: arith.extf

// -----

func.func @extf_vector_f8E8M0FNU_to_f16(%arg0 : vector<4xf8E8M0FNU>) -> vector<4xf16> {
    %0 = arith.extf %arg0 : vector<4xf8E8M0FNU> to vector<4xf16>
    return %0 : vector<4xf16>
}

// CHECK-LABEL: @extf_vector_f8E8M0FNU_to_f16
// CHECK-NOT: arith.extf

// -----

func.func @extf_vector_f8E8M0FNU_to_bf16(%arg0 : vector<4xf8E8M0FNU>) -> vector<4xbf16> {
    %0 = arith.extf %arg0 : vector<4xf8E8M0FNU> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @extf_vector_f8E8M0FNU_to_bf16
// CHECK-NOT: arith.extf
// CHECK: return

// -----

func.func @scaling_extf_to_f32(%arg0: f4E2M1FN, %arg1 : f8E8M0FNU) -> f32 {
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f8E8M0FNU to f32
    return %0 : f32 
}

// SCHECK-LABEL: @scaling_extf_to_f32
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : f4E2M1FN to f32
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : f32
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_to_f32_using_f16_scales(%arg0: f4E2M1FN, %arg1 : f16) -> f32 {
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f16 to f32
    return %0 : f32 
}

// SCHECK-LABEL: @scaling_extf_to_f32_using_f16_scales
// SCHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 : f16 to f8E8M0FNU
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] : f8E8M0FNU to f32
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : f4E2M1FN to f32
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : f32
// SCHECK: return %[[RESULT]]

// -----

func.func @invalid_scaling_extf_to_f32(%arg0: f4E2M1FN, %arg1 : f8E5M2FNUZ) -> f32 {
    // expected-error@+1 {{failed to legalize operation 'arith.scaling_extf' that was explicitly marked illegal}}
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f8E5M2FNUZ to f32
    return %0 : f32
}

// -----

func.func @scaling_extf_vector_to_f32(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xf32>
    return %0 : vector<4xf32>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f32
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf32>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf32> 
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f16(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xf16> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xf16>
    return %0 : vector<4xf16>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f16
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf16>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf16>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf16> 
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_bf16(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xbf16> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// SCHECK-LABEL: @scaling_extf_vector_to_bf16
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xbf16>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xbf16>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xbf16> 
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f32_using_f16_scales(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf16>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf16> to vector<4xf32>
    return %0 : vector<4xf32>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f32_using_f16_scales
// SCHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] : vector<4xf8E8M0FNU> to vector<4xf32>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf32>
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f32_using_f16_scales_fastmath(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf16>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 fastmath<fast> : vector<4xf4E2M1FN>, vector<4xf16> to vector<4xf32>
    return %0 : vector<4xf32>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f32_using_f16_scales_fastmath
// SCHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 fastmath<fast> : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] fastmath<fast> : vector<4xf8E8M0FNU> to vector<4xf32>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 fastmath<fast> : vector<4xf4E2M1FN> to vector<4xf32>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] fastmath<fast> : vector<4xf32>
// SCHECK: return %[[RESULT]]

// -----

func.func @maxsi(%a: i32, %b: i32) -> i32 {
  %result = arith.maxsi %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @maxsi
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi sgt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @minsi(%a: i32, %b: i32) -> i32 {
  %result = arith.minsi %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @minsi
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @maxui(%a: i32, %b: i32) -> i32 {
  %result = arith.maxui %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @maxui
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ugt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @minui(%a: i32, %b: i32) -> i32 {
  %result = arith.minui %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @minui
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @truncf_f32_to_f4E2M1FN(%arg0 : f32) -> f4E2M1FN {
    %0 = arith.truncf %arg0 : f32 to f4E2M1FN
    return %0 : f4E2M1FN
}

// CHECK-LABEL: @truncf_f32_to_f4E2M1FN
// CHECK-NOT: arith.truncf

// -----

func.func @truncf_vector_f32_to_f4E2M1FN(%arg0 : vector<4xf32>) -> vector<4xf4E2M1FN> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf4E2M1FN>
    return %0 : vector<4xf4E2M1FN>
}

// CHECK-LABEL: @truncf_vector_f32_to_f4E2M1FN
// CHECK-NOT: arith.truncf

// -----

func.func @extf_f4E2M1FN_to_f32(%arg0 : f4E2M1FN) -> f32 {
    %0 = arith.extf %arg0 : f4E2M1FN to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f4E2M1FN_to_f32
// CHECK-NOT: arith.extf

// -----

func.func @extf_vector_f4E2M1FN_to_f32(%arg0 : vector<4xf4E2M1FN>) -> vector<4xf32> {
    %0 = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @extf_vector_f4E2M1FN_to_f32
// CHECK-NOT: arith.extf
