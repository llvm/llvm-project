// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: test_addi
func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.addi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_addi_tensor
func.func @test_addi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.addi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_addi_vector
func.func @test_addi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.addi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_addi_scalable_vector
func.func @test_addi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.addi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_addui_extended
func.func @test_addui_extended(%arg0 : i64, %arg1 : i64) -> i64 {
  %sum, %overflow = arith.addui_extended %arg0, %arg1 : i64, i1
  return %sum : i64
}

// CHECK-LABEL: test_addui_extended_tensor
func.func @test_addui_extended_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %sum, %overflow = arith.addui_extended %arg0, %arg1 : tensor<8x8xi64>, tensor<8x8xi1>
  return %sum : tensor<8x8xi64>
}

// CHECK-LABEL: test_addui_extended_vector
func.func @test_addui_extended_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0:2 = arith.addui_extended %arg0, %arg1 : vector<8xi64>, vector<8xi1>
  return %0#0 : vector<8xi64>
}

// CHECK-LABEL: test_addui_extended_scalable_vector
func.func @test_addui_extended_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0:2 = arith.addui_extended %arg0, %arg1 : vector<[8]xi64>, vector<[8]xi1>
  return %0#0 : vector<[8]xi64>
}

// CHECK-LABEL: test_subi
func.func @test_subi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.subi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_subi_tensor
func.func @test_subi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.subi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_subi_vector
func.func @test_subi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.subi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_subi_scalable_vector
func.func @test_subi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.subi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_muli
func.func @test_muli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.muli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_muli_tensor
func.func @test_muli_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.muli %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_muli_vector
func.func @test_muli_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.muli %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_muli_scalable_vector
func.func @test_muli_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.muli %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_mulsi_extended
func.func @test_mulsi_extended(%arg0 : i32, %arg1 : i32) -> i32 {
  %low, %high = arith.mulsi_extended %arg0, %arg1 : i32
  return %high : i32
}

// CHECK-LABEL: test_mulsi_extended_tensor
func.func @test_mulsi_extended_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %low, %high = arith.mulsi_extended %arg0, %arg1 : tensor<8x8xi64>
  return %high : tensor<8x8xi64>
}

// CHECK-LABEL: test_mulsi_extended_vector
func.func @test_mulsi_extended_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0:2 = arith.mulsi_extended %arg0, %arg1 : vector<8xi64>
  return %0#0 : vector<8xi64>
}

// CHECK-LABEL: test_mulsi_extended_scalable_vector
func.func @test_mulsi_extended_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0:2 = arith.mulsi_extended %arg0, %arg1 : vector<[8]xi64>
  return %0#1 : vector<[8]xi64>
}

// CHECK-LABEL: test_mului_extended
func.func @test_mului_extended(%arg0 : i32, %arg1 : i32) -> i32 {
  %low, %high = arith.mului_extended %arg0, %arg1 : i32
  return %high : i32
}

// CHECK-LABEL: test_mului_extended_tensor
func.func @test_mului_extended_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %low, %high = arith.mului_extended %arg0, %arg1 : tensor<8x8xi64>
  return %high : tensor<8x8xi64>
}

// CHECK-LABEL: test_mului_extended_vector
func.func @test_mului_extended_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0:2 = arith.mului_extended %arg0, %arg1 : vector<8xi64>
  return %0#0 : vector<8xi64>
}

// CHECK-LABEL: test_mului_extended_scalable_vector
func.func @test_mului_extended_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0:2 = arith.mului_extended %arg0, %arg1 : vector<[8]xi64>
  return %0#1 : vector<[8]xi64>
}

// CHECK-LABEL: test_divui
func.func @test_divui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.divui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_divui_tensor
func.func @test_divui_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.divui %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_divui_vector
func.func @test_divui_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.divui %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_divui_scalable_vector
func.func @test_divui_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.divui %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_divsi
func.func @test_divsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.divsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_divsi_tensor
func.func @test_divsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.divsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_divsi_vector
func.func @test_divsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.divsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_divsi_scalable_vector
func.func @test_divsi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.divsi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_remui
func.func @test_remui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.remui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_remui_tensor
func.func @test_remui_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.remui %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_remui_vector
func.func @test_remui_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.remui %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_remui_scalable_vector
func.func @test_remui_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.remui %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_remsi
func.func @test_remsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.remsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_remsi_tensor
func.func @test_remsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.remsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_remsi_vector
func.func @test_remsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.remsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_remsi_scalable_vector
func.func @test_remsi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.remsi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_andi
func.func @test_andi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.andi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_andi_tensor
func.func @test_andi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.andi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_andi_vector
func.func @test_andi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.andi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_andi_scalable_vector
func.func @test_andi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.andi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_ori
func.func @test_ori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.ori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_ori_tensor
func.func @test_ori_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.ori %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_ori_vector
func.func @test_ori_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.ori %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_ori_scalable_vector
func.func @test_ori_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.ori %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_xori
func.func @test_xori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.xori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_xori_tensor
func.func @test_xori_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.xori %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_xori_vector
func.func @test_xori_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.xori %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_xori_scalable_vector
func.func @test_xori_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.xori %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_ceildivsi
func.func @test_ceildivsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.ceildivsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_ceildivsi_tensor
func.func @test_ceildivsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.ceildivsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_ceildivsi_vector
func.func @test_ceildivsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.ceildivsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_ceildivsi_scalable_vector
func.func @test_ceildivsi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.ceildivsi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_floordivsi
func.func @test_floordivsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.floordivsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_floordivsi_tensor
func.func @test_floordivsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.floordivsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_floordivsi_vector
func.func @test_floordivsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.floordivsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_floordivsi_scalable_vector
func.func @test_floordivsi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.floordivsi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_shli
func.func @test_shli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shli_tensor
func.func @test_shli_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.shli %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_shli_vector
func.func @test_shli_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.shli %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_shli_scalable_vector
func.func @test_shli_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.shli %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_shrui
func.func @test_shrui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrui_tensor
func.func @test_shrui_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.shrui %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_shrui_vector
func.func @test_shrui_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.shrui %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_shrui_scalable_vector
func.func @test_shrui_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.shrui %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_shrsi
func.func @test_shrsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrsi_tensor
func.func @test_shrsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.shrsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_shrsi_vector
func.func @test_shrsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.shrsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_shrsi_scalable_vector
func.func @test_shrsi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi64> {
  %0 = arith.shrsi %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_negf
func.func @test_negf(%arg0 : f64) -> f64 {
  %0 = arith.negf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: test_negf_tensor
func.func @test_negf_tensor(%arg0 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.negf %arg0 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_negf_vector
func.func @test_negf_vector(%arg0 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.negf %arg0 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_negf_scalable_vector
func.func @test_negf_scalable_vector(%arg0 : vector<[8]xf64>) -> vector<[8]xf64> {
  %0 = arith.negf %arg0 : vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_addf
func.func @test_addf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.addf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_addf_tensor
func.func @test_addf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.addf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_addf_vector
func.func @test_addf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.addf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_addf_scalable_vector
func.func @test_addf_scalable_vector(%arg0 : vector<[8]xf64>, %arg1 : vector<[8]xf64>) -> vector<[8]xf64> {
  %0 = arith.addf %arg0, %arg1 : vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_subf
func.func @test_subf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.subf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_subf_tensor
func.func @test_subf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.subf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_subf_vector
func.func @test_subf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.subf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_subf_scalable_vector
func.func @test_subf_scalable_vector(%arg0 : vector<[8]xf64>, %arg1 : vector<[8]xf64>) -> vector<[8]xf64> {
  %0 = arith.subf %arg0, %arg1 : vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_mulf
func.func @test_mulf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.mulf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_mulf_tensor
func.func @test_mulf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.mulf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_mulf_vector
func.func @test_mulf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.mulf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_mulf_scalable_vector
func.func @test_mulf_scalable_vector(%arg0 : vector<[8]xf64>, %arg1 : vector<[8]xf64>) -> vector<[8]xf64> {
  %0 = arith.mulf %arg0, %arg1 : vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_divf
func.func @test_divf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_divf_tensor
func.func @test_divf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.divf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_divf_vector
func.func @test_divf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.divf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_divf_scalable_vector
func.func @test_divf_scalable_vector(%arg0 : vector<[8]xf64>, %arg1 : vector<[8]xf64>) -> vector<[8]xf64> {
  %0 = arith.divf %arg0, %arg1 : vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_remf
func.func @test_remf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.remf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_remf_tensor
func.func @test_remf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.remf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_remf_vector
func.func @test_remf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.remf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_remf_scalable_vector
func.func @test_remf_scalable_vector(%arg0 : vector<[8]xf64>, %arg1 : vector<[8]xf64>) -> vector<[8]xf64> {
  %0 = arith.remf %arg0, %arg1 : vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_extui
func.func @test_extui(%arg0 : i32) -> i64 {
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: test_extui_tensor
func.func @test_extui_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xi64> {
  %0 = arith.extui %arg0 : tensor<8x8xi32> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_extui_vector
func.func @test_extui_vector(%arg0 : vector<8xi32>) -> vector<8xi64> {
  %0 = arith.extui %arg0 : vector<8xi32> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_extui_scalable_vector
func.func @test_extui_scalable_vector(%arg0 : vector<[8]xi32>) -> vector<[8]xi64> {
  %0 = arith.extui %arg0 : vector<[8]xi32> to vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_extsi
func.func @test_extsi(%arg0 : i32) -> i64 {
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: test_extsi_tensor
func.func @test_extsi_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xi64> {
  %0 = arith.extsi %arg0 : tensor<8x8xi32> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_extsi_vector
func.func @test_extsi_vector(%arg0 : vector<8xi32>) -> vector<8xi64> {
  %0 = arith.extsi %arg0 : vector<8xi32> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_extsi_scalable_vector
func.func @test_extsi_scalable_vector(%arg0 : vector<[8]xi32>) -> vector<[8]xi64> {
  %0 = arith.extsi %arg0 : vector<[8]xi32> to vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_extf
func.func @test_extf(%arg0 : f32) -> f64 {
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: test_extf_tensor
func.func @test_extf_tensor(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf64> {
  %0 = arith.extf %arg0 : tensor<8x8xf32> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_extf_tensor_encoding
func.func @test_extf_tensor_encoding(%arg0 : tensor<8x8xf32, "foo">) -> tensor<8x8xf64, "foo"> {
  %0 = arith.extf %arg0 : tensor<8x8xf32, "foo"> to tensor<8x8xf64, "foo">
  return %0 : tensor<8x8xf64, "foo">
}

// CHECK-LABEL: test_extf_vector
func.func @test_extf_vector(%arg0 : vector<8xf32>) -> vector<8xf64> {
  %0 = arith.extf %arg0 : vector<8xf32> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_extf_scalable_vector
func.func @test_extf_scalable_vector(%arg0 : vector<[8]xf32>) -> vector<[8]xf64> {
  %0 = arith.extf %arg0 : vector<[8]xf32> to vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_trunci
func.func @test_trunci(%arg0 : i32) -> i16 {
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}

// CHECK-LABEL: test_trunci_tensor
func.func @test_trunci_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xi16> {
  %0 = arith.trunci %arg0 : tensor<8x8xi32> to tensor<8x8xi16>
  return %0 : tensor<8x8xi16>
}

// CHECK-LABEL: test_trunci_vector
func.func @test_trunci_vector(%arg0 : vector<8xi32>) -> vector<8xi16> {
  %0 = arith.trunci %arg0 : vector<8xi32> to vector<8xi16>
  return %0 : vector<8xi16>
}

// CHECK-LABEL: test_trunci_scalable_vector
func.func @test_trunci_scalable_vector(%arg0 : vector<[8]xi32>) -> vector<[8]xi16> {
  %0 = arith.trunci %arg0 : vector<[8]xi32> to vector<[8]xi16>
  return %0 : vector<[8]xi16>
}

// CHECK-LABEL: test_truncf
func.func @test_truncf(%arg0 : f32) -> bf16 {
  %0 = arith.truncf %arg0 : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: test_truncf_tensor
func.func @test_truncf_tensor(%arg0 : tensor<8x8xf32>) -> tensor<8x8xbf16> {
  %0 = arith.truncf %arg0 : tensor<8x8xf32> to tensor<8x8xbf16>
  return %0 : tensor<8x8xbf16>
}

// CHECK-LABEL: test_truncf_vector
func.func @test_truncf_vector(%arg0 : vector<8xf32>) -> vector<8xbf16> {
  %0 = arith.truncf %arg0 : vector<8xf32> to vector<8xbf16>
  return %0 : vector<8xbf16>
}

// CHECK-LABEL: test_truncf_scalable_vector
func.func @test_truncf_scalable_vector(%arg0 : vector<[8]xf32>) -> vector<[8]xbf16> {
  %0 = arith.truncf %arg0 : vector<[8]xf32> to vector<[8]xbf16>
  return %0 : vector<[8]xbf16>
}

// CHECK-LABEL: test_truncf_rounding_mode
func.func @test_truncf_rounding_mode(%arg0 : f64) -> (f32, f32, f32, f32, f32) {
  %0 = arith.truncf %arg0 tonearesteven : f64 to f32
  %1 = arith.truncf %arg0 downward : f64 to f32
  %2 = arith.truncf %arg0 upward : f64 to f32
  %3 = arith.truncf %arg0 towardzero : f64 to f32
  %4 = arith.truncf %arg0 tonearestaway : f64 to f32
  return %0, %1, %2, %3, %4 : f32, f32, f32, f32, f32
}

// CHECK-LABEL: test_uitofp
func.func @test_uitofp(%arg0 : i32) -> f32 {
  %0 = arith.uitofp %arg0 : i32 to f32
 return %0 : f32
}

// CHECK-LABEL: test_uitofp_tensor
func.func @test_uitofp_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xf32> {
  %0 = arith.uitofp %arg0 : tensor<8x8xi32> to tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: test_uitofp_vector
func.func @test_uitofp_vector(%arg0 : vector<8xi32>) -> vector<8xf32> {
  %0 = arith.uitofp %arg0 : vector<8xi32> to vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: test_uitofp_scalable_vector
func.func @test_uitofp_scalable_vector(%arg0 : vector<[8]xi32>) -> vector<[8]xf32> {
  %0 = arith.uitofp %arg0 : vector<[8]xi32> to vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// CHECK-LABEL: test_sitofp
func.func @test_sitofp(%arg0 : i16) -> f64 {
  %0 = arith.sitofp %arg0 : i16 to f64
  return %0 : f64
}

// CHECK-LABEL: test_sitofp_tensor
func.func @test_sitofp_tensor(%arg0 : tensor<8x8xi16>) -> tensor<8x8xf64> {
  %0 = arith.sitofp %arg0 : tensor<8x8xi16> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_sitofp_vector
func.func @test_sitofp_vector(%arg0 : vector<8xi16>) -> vector<8xf64> {
  %0 = arith.sitofp %arg0 : vector<8xi16> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_sitofp_scalable_vector
func.func @test_sitofp_scalable_vector(%arg0 : vector<[8]xi16>) -> vector<[8]xf64> {
  %0 = arith.sitofp %arg0 : vector<[8]xi16> to vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_fptoui
func.func @test_fptoui(%arg0 : bf16) -> i8 {
  %0 = arith.fptoui %arg0 : bf16 to i8
  return %0 : i8
}

// CHECK-LABEL: test_fptoui_tensor
func.func @test_fptoui_tensor(%arg0 : tensor<8x8xbf16>) -> tensor<8x8xi8> {
  %0 = arith.fptoui %arg0 : tensor<8x8xbf16> to tensor<8x8xi8>
  return %0 : tensor<8x8xi8>
}

// CHECK-LABEL: test_fptoui_vector
func.func @test_fptoui_vector(%arg0 : vector<8xbf16>) -> vector<8xi8> {
  %0 = arith.fptoui %arg0 : vector<8xbf16> to vector<8xi8>
 return %0 : vector<8xi8>
}

// CHECK-LABEL: test_fptoui_scalable_vector
func.func @test_fptoui_scalable_vector(%arg0 : vector<[8]xbf16>) -> vector<[8]xi8> {
  %0 = arith.fptoui %arg0 : vector<[8]xbf16> to vector<[8]xi8>
  return %0 : vector<[8]xi8>
}

// CHECK-LABEL: test_fptosi
func.func @test_fptosi(%arg0 : f64) -> i64 {
  %0 = arith.fptosi %arg0 : f64 to i64
  return %0 : i64
}

// CHECK-LABEL: test_fptosi_tensor
func.func @test_fptosi_tensor(%arg0 : tensor<8x8xf64>) -> tensor<8x8xi64> {
  %0 = arith.fptosi %arg0 : tensor<8x8xf64> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_fptosi_vector
func.func @test_fptosi_vector(%arg0 : vector<8xf64>) -> vector<8xi64> {
  %0 = arith.fptosi %arg0 : vector<8xf64> to vector<8xi64>
 return %0 : vector<8xi64>
}

// CHECK-LABEL: test_fptosi_scalable_vector
func.func @test_fptosi_scalable_vector(%arg0 : vector<[8]xf64>) -> vector<[8]xi64> {
  %0 = arith.fptosi %arg0 : vector<[8]xf64> to vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_index_cast0
func.func @test_index_cast0(%arg0 : i32) -> index {
  %0 = arith.index_cast %arg0 : i32 to index
  return %0 : index
}

// CHECK-LABEL: test_index_cast_tensor0
func.func @test_index_cast_tensor0(%arg0 : tensor<8x8xi32>) -> tensor<8x8xindex> {
  %0 = arith.index_cast %arg0 : tensor<8x8xi32> to tensor<8x8xindex>
  return %0 : tensor<8x8xindex>
}

// CHECK-LABEL: test_index_cast_vector0
func.func @test_index_cast_vector0(%arg0 : vector<8xi32>) -> vector<8xindex> {
  %0 = arith.index_cast %arg0 : vector<8xi32> to vector<8xindex>
  return %0 : vector<8xindex>
}

// CHECK-LABEL: test_index_cast_scalable_vector0
func.func @test_index_cast_scalable_vector0(%arg0 : vector<[8]xi32>) -> vector<[8]xindex> {
  %0 = arith.index_cast %arg0 : vector<[8]xi32> to vector<[8]xindex>
  return %0 : vector<[8]xindex>
}

// CHECK-LABEL: test_index_cast1
func.func @test_index_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_cast_tensor1
func.func @test_index_cast_tensor1(%arg0 : tensor<8x8xindex>) -> tensor<8x8xi64> {
  %0 = arith.index_cast %arg0 : tensor<8x8xindex> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_index_cast_vector1
func.func @test_index_cast_vector1(%arg0 : vector<8xindex>) -> vector<8xi64> {
  %0 = arith.index_cast %arg0 : vector<8xindex> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_index_cast_scalable_vector1
func.func @test_index_cast_scalable_vector1(%arg0 : vector<[8]xindex>) -> vector<[8]xi64> {
  %0 = arith.index_cast %arg0 : vector<[8]xindex> to vector<[8]xi64>
  return %0 : vector<[8]xi64>
}


// CHECK-LABEL: test_index_castui0
func.func @test_index_castui0(%arg0 : i32) -> index {
  %0 = arith.index_castui %arg0 : i32 to index
  return %0 : index
}

// CHECK-LABEL: test_index_castui_tensor0
func.func @test_index_castui_tensor0(%arg0 : tensor<8x8xi32>) -> tensor<8x8xindex> {
  %0 = arith.index_castui %arg0 : tensor<8x8xi32> to tensor<8x8xindex>
  return %0 : tensor<8x8xindex>
}

// CHECK-LABEL: test_index_castui_vector0
func.func @test_index_castui_vector0(%arg0 : vector<8xi32>) -> vector<8xindex> {
  %0 = arith.index_castui %arg0 : vector<8xi32> to vector<8xindex>
  return %0 : vector<8xindex>
}

// CHECK-LABEL: test_index_castui_scalable_vector0
func.func @test_index_castui_scalable_vector0(%arg0 : vector<[8]xi32>) -> vector<[8]xindex> {
  %0 = arith.index_castui %arg0 : vector<[8]xi32> to vector<[8]xindex>
  return %0 : vector<[8]xindex>
}

// CHECK-LABEL: test_indexui_cast1
func.func @test_indexui_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_castui %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_castui_tensor1
func.func @test_index_castui_tensor1(%arg0 : tensor<8x8xindex>) -> tensor<8x8xi64> {
  %0 = arith.index_castui %arg0 : tensor<8x8xindex> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_index_castui_vector1
func.func @test_index_castui_vector1(%arg0 : vector<8xindex>) -> vector<8xi64> {
  %0 = arith.index_castui %arg0 : vector<8xindex> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_index_castui_scalable_vector1
func.func @test_index_castui_scalable_vector1(%arg0 : vector<[8]xindex>) -> vector<[8]xi64> {
  %0 = arith.index_castui %arg0 : vector<[8]xindex> to vector<[8]xi64>
  return %0 : vector<[8]xi64>
}

// CHECK-LABEL: test_bitcast0
func.func @test_bitcast0(%arg0 : i64) -> f64 {
  %0 = arith.bitcast %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: test_bitcast_tensor0
func.func @test_bitcast_tensor0(%arg0 : tensor<8x8xi64>) -> tensor<8x8xf64> {
  %0 = arith.bitcast %arg0 : tensor<8x8xi64> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_bitcast_vector0
func.func @test_bitcast_vector0(%arg0 : vector<8xi64>) -> vector<8xf64> {
  %0 = arith.bitcast %arg0 : vector<8xi64> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_bitcast_scalable_vector0
func.func @test_bitcast_scalable_vector0(%arg0 : vector<[8]xi64>) -> vector<[8]xf64> {
  %0 = arith.bitcast %arg0 : vector<[8]xi64> to vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_bitcast1
func.func @test_bitcast1(%arg0 : f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: test_bitcast_tensor1
func.func @test_bitcast_tensor1(%arg0 : tensor<8x8xf32>) -> tensor<8x8xi32> {
  %0 = arith.bitcast %arg0 : tensor<8x8xf32> to tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: test_bitcast_vector1
func.func @test_bitcast_vector1(%arg0 : vector<8xf32>) -> vector<8xi32> {
  %0 = arith.bitcast %arg0 : vector<8xf32> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: test_bitcast_scalable_vector1
func.func @test_bitcast_scalable_vector1(%arg0 : vector<[8]xf32>) -> vector<[8]xi32> {
  %0 = arith.bitcast %arg0 : vector<[8]xf32> to vector<[8]xi32>
  return %0 : vector<[8]xi32>
}

// CHECK-LABEL: test_cmpi
func.func @test_cmpi(%arg0 : i64, %arg1 : i64) -> i1 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  return %0 : i1
}

// CHECK-LABEL: test_cmpi_tensor
func.func @test_cmpi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi1> {
  %0 = arith.cmpi slt, %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi1>
}

// CHECK-LABEL: test_cmpi_tensor_encoding
func.func @test_cmpi_tensor_encoding(%arg0 : tensor<8x8xi64, "foo">, %arg1 : tensor<8x8xi64, "foo">) -> tensor<8x8xi1, "foo"> {
  %0 = arith.cmpi slt, %arg0, %arg1 : tensor<8x8xi64, "foo">
  return %0 : tensor<8x8xi1, "foo">
}

// CHECK-LABEL: test_cmpi_vector
func.func @test_cmpi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi1> {
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: test_cmpi_scalable_vector
func.func @test_cmpi_scalable_vector(%arg0 : vector<[8]xi64>, %arg1 : vector<[8]xi64>) -> vector<[8]xi1> {
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<[8]xi64>
  return %0 : vector<[8]xi1>
}

// CHECK-LABEL: test_cmpi_vector_0d
func.func @test_cmpi_vector_0d(%arg0 : vector<i64>, %arg1 : vector<i64>) -> vector<i1> {
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<i64>
  return %0 : vector<i1>
}

// CHECK-LABEL: test_cmpf
func.func @test_cmpf(%arg0 : f64, %arg1 : f64) -> i1 {
  %0 = arith.cmpf oeq, %arg0, %arg1 : f64
  return %0 : i1
}

// CHECK-LABEL: test_cmpf_tensor
func.func @test_cmpf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xi1> {
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xi1>
}

// CHECK-LABEL: test_cmpf_vector
func.func @test_cmpf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xi1> {
  %0 = arith.cmpf ult, %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: test_cmpf_scalable_vector
func.func @test_cmpf_scalable_vector(%arg0 : vector<[8]xf64>, %arg1 : vector<[8]xf64>) -> vector<[8]xi1> {
  %0 = arith.cmpf ult, %arg0, %arg1 : vector<[8]xf64>
  return %0 : vector<[8]xi1>
}

// CHECK-LABEL: test_index_cast
func.func @test_index_cast(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_cast_tensor
func.func @test_index_cast_tensor(%arg0 : tensor<index>) -> tensor<i64> {
  %0 = arith.index_cast %arg0 : tensor<index> to tensor<i64>
  return %0 : tensor<i64>
}

// CHECK-LABEL: test_index_cast_tensor_reverse
func.func @test_index_cast_tensor_reverse(%arg0 : tensor<i64>) -> tensor<index> {
  %0 = arith.index_cast %arg0 : tensor<i64> to tensor<index>
  return %0 : tensor<index>
}

// CHECK-LABEL: func @bitcast(
func.func @bitcast(%arg : f32) -> i32 {
  %res = arith.bitcast %arg : f32 to i32
  return %res : i32
}

// CHECK-LABEL: test_constant
func.func @test_constant() -> () {
  // CHECK: %c42_i32 = arith.constant 42 : i32
  %0 = "arith.constant"(){value = 42 : i32} : () -> i32

  // CHECK: %c42_i32_0 = arith.constant 42 : i32
  %1 = arith.constant 42 : i32

  // CHECK: %c43 = arith.constant {crazy = "func.foo"} 43 : index
  %2 = arith.constant {crazy = "func.foo"} 43: index

  // CHECK: %cst = arith.constant 4.300000e+01 : bf16
  %3 = arith.constant 43.0 : bf16

  // CHECK: %cst_1 = arith.constant dense<0> : vector<4xi32>
  %4 = arith.constant dense<0> : vector<4 x i32>

  // CHECK: %cst_2 = arith.constant dense<0> : tensor<42xi32>
  %5 = arith.constant dense<0> : tensor<42 x i32>

  // CHECK: %cst_3 = arith.constant dense<0> : vector<42xi32>
  %6 = arith.constant dense<0> : vector<42 x i32>

  // CHECK: %true = arith.constant true
  %7 = arith.constant true

  // CHECK: %false = arith.constant false
  %8 = arith.constant false

  // CHECK: %c-1_i128 = arith.constant -1 : i128
  %9 = arith.constant 340282366920938463463374607431768211455 : i128

  // CHECK: %c85070591730234615865843651857942052864_i128 = arith.constant 85070591730234615865843651857942052864 : i128
  %10 = arith.constant 85070591730234615865843651857942052864 : i128

  return
}

// CHECK-LABEL: func @maximum
func.func @maximum(%v1: vector<4xf32>, %v2: vector<4xf32>,
               %sv1: vector<[4]xf32>, %sv2: vector<[4]xf32>,
               %f1: f32, %f2: f32,
               %i1: i32, %i2: i32) {
  %maximum_vector = arith.maximumf %v1, %v2 : vector<4xf32>
  %maximum_scalable_vector = arith.maximumf %sv1, %sv2 : vector<[4]xf32>
  %maximum_float = arith.maximumf %f1, %f2 : f32
  %maxnum_vector = arith.maxnumf %v1, %v2 : vector<4xf32>
  %maxnum_scalable_vector = arith.maxnumf %sv1, %sv2 : vector<[4]xf32>
  %maxnum_float = arith.maxnumf %f1, %f2 : f32
  %max_signed = arith.maxsi %i1, %i2 : i32
  %max_unsigned = arith.maxui %i1, %i2 : i32
  return
}

// CHECK-LABEL: func @minimum
func.func @minimum(%v1: vector<4xf32>, %v2: vector<4xf32>,
               %sv1: vector<[4]xf32>, %sv2: vector<[4]xf32>,
               %f1: f32, %f2: f32,
               %i1: i32, %i2: i32) {
  %minimum_vector = arith.minimumf %v1, %v2 : vector<4xf32>
  %minimum_scalable_vector = arith.minimumf %sv1, %sv2 : vector<[4]xf32>
  %minimum_float = arith.minimumf %f1, %f2 : f32
  %minnum_vector = arith.minnumf %v1, %v2 : vector<4xf32>
  %minnum_scalable_vector = arith.minnumf %sv1, %sv2 : vector<[4]xf32>
  %minnum_float = arith.minnumf %f1, %f2 : f32
  %min_signed = arith.minsi %i1, %i2 : i32
  %min_unsigned = arith.minui %i1, %i2 : i32
  return
}

// CHECK-LABEL: @fastmath
func.func @fastmath(%arg0: f32, %arg1: f32, %arg2: i32) {
// CHECK: {{.*}} = arith.addf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.subf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.mulf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.divf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.remf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.negf %arg0 fastmath<fast> : f32
  %0 = arith.addf %arg0, %arg1 fastmath<fast> : f32
  %1 = arith.subf %arg0, %arg1 fastmath<fast> : f32
  %2 = arith.mulf %arg0, %arg1 fastmath<fast> : f32
  %3 = arith.divf %arg0, %arg1 fastmath<fast> : f32
  %4 = arith.remf %arg0, %arg1 fastmath<fast> : f32
  %5 = arith.negf %arg0 fastmath<fast> : f32
// CHECK: {{.*}} = arith.addf %arg0, %arg1 : f32
  %6 = arith.addf %arg0, %arg1 fastmath<none> : f32
// CHECK: {{.*}} = arith.addf %arg0, %arg1 fastmath<nnan,ninf> : f32
  %7 = arith.addf %arg0, %arg1 fastmath<nnan,ninf> : f32
// CHECK: {{.*}} = arith.mulf %arg0, %arg1 fastmath<fast> : f32
  %8 = arith.mulf %arg0, %arg1 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f32
// CHECK: {{.*}} = arith.cmpf oeq, %arg0, %arg1 fastmath<fast> : f32
  %9 = arith.cmpf oeq, %arg0, %arg1 fastmath<fast> : f32

  return
}

// CHECK-LABEL: @select_tensor
func.func @select_tensor(%arg0 : tensor<8xi1>, %arg1 : tensor<8xi32>, %arg2 : tensor<8xi32>) -> tensor<8xi32> {
  // CHECK: = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xi1>, tensor<8xi32>
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<8xi1>, tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: @select_tensor_encoding
func.func @select_tensor_encoding(
  %arg0 : tensor<8xi1, "foo">, %arg1 : tensor<8xi32, "foo">, %arg2 : tensor<8xi32, "foo">) -> tensor<8xi32, "foo"> {
  // CHECK: = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xi1, "foo">, tensor<8xi32, "foo">
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<8xi1, "foo">, tensor<8xi32, "foo">
  return %0 : tensor<8xi32, "foo">
}

// CHECK-LABEL: @intflags_func
func.func @intflags_func(%arg0: i64, %arg1: i64) {
  // CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} overflow<nsw> : i64
  %0 = arith.addi %arg0, %arg1 overflow<nsw> : i64
  // CHECK: %{{.*}} = arith.subi %{{.*}}, %{{.*}} overflow<nuw> : i64
  %1 = arith.subi %arg0, %arg1 overflow<nuw> : i64
  // CHECK: %{{.*}} = arith.muli %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
  %2 = arith.muli %arg0, %arg1 overflow<nsw, nuw> : i64
  // CHECK: %{{.*}} = arith.shli %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
  %3 = arith.shli %arg0, %arg1 overflow<nsw, nuw> : i64
  return
}
