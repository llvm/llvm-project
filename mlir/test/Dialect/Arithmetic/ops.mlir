// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: test_addi
func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.addi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_addi_tensor
func @test_addi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.addi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_addi_vector
func @test_addi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.addi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_subi
func @test_subi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.subi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_subi_tensor
func @test_subi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.subi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_subi_vector
func @test_subi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.subi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_muli
func @test_muli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.muli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_muli_tensor
func @test_muli_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.muli %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_muli_vector
func @test_muli_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.muli %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_divui
func @test_divui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.divui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_divui_tensor
func @test_divui_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.divui %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_divui_vector
func @test_divui_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.divui %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_divsi
func @test_divsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.divsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_divsi_tensor
func @test_divsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.divsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_divsi_vector
func @test_divsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.divsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_remui
func @test_remui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.remui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_remui_tensor
func @test_remui_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.remui %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_remui_vector
func @test_remui_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.remui %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_remsi
func @test_remsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.remsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_remsi_tensor
func @test_remsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.remsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_remsi_vector
func @test_remsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.remsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_andi
func @test_andi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.andi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_andi_tensor
func @test_andi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.andi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_andi_vector
func @test_andi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.andi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_ori
func @test_ori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.ori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_ori_tensor
func @test_ori_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.ori %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_ori_vector
func @test_ori_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.ori %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_xori
func @test_xori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.xori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_xori_tensor
func @test_xori_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.xori %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_xori_vector
func @test_xori_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.xori %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_ceildivsi
func @test_ceildivsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.ceildivsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_ceildivsi_tensor
func @test_ceildivsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.ceildivsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_ceildivsi_vector
func @test_ceildivsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.ceildivsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_floordivsi
func @test_floordivsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.floordivsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_floordivsi_tensor
func @test_floordivsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.floordivsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_floordivsi_vector
func @test_floordivsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.floordivsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_shli
func @test_shli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shli_tensor
func @test_shli_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.shli %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_shli_vector
func @test_shli_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.shli %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_shrui
func @test_shrui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrui_tensor
func @test_shrui_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.shrui %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_shrui_vector
func @test_shrui_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.shrui %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_shrsi
func @test_shrsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrsi_tensor
func @test_shrsi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi64> {
  %0 = arith.shrsi %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_shrsi_vector
func @test_shrsi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi64> {
  %0 = arith.shrsi %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_negf
func @test_negf(%arg0 : f64) -> f64 {
  %0 = arith.negf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: test_negf_tensor
func @test_negf_tensor(%arg0 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.negf %arg0 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_negf_vector
func @test_negf_vector(%arg0 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.negf %arg0 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_addf
func @test_addf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.addf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_addf_tensor
func @test_addf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.addf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_addf_vector
func @test_addf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.addf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_subf
func @test_subf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.subf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_subf_tensor
func @test_subf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.subf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_subf_vector
func @test_subf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.subf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_mulf
func @test_mulf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.mulf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_mulf_tensor
func @test_mulf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.mulf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_mulf_vector
func @test_mulf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.mulf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_divf
func @test_divf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_divf_tensor
func @test_divf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.divf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_divf_vector
func @test_divf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.divf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_remf
func @test_remf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.remf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_remf_tensor
func @test_remf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = arith.remf %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_remf_vector
func @test_remf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xf64> {
  %0 = arith.remf %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_extui
func @test_extui(%arg0 : i32) -> i64 {
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: test_extui_tensor
func @test_extui_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xi64> {
  %0 = arith.extui %arg0 : tensor<8x8xi32> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_extui_vector
func @test_extui_vector(%arg0 : vector<8xi32>) -> vector<8xi64> {
  %0 = arith.extui %arg0 : vector<8xi32> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_extsi
func @test_extsi(%arg0 : i32) -> i64 {
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: test_extsi_tensor
func @test_extsi_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xi64> {
  %0 = arith.extsi %arg0 : tensor<8x8xi32> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_extsi_vector
func @test_extsi_vector(%arg0 : vector<8xi32>) -> vector<8xi64> {
  %0 = arith.extsi %arg0 : vector<8xi32> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_extf
func @test_extf(%arg0 : f32) -> f64 {
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: test_extf_tensor
func @test_extf_tensor(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf64> {
  %0 = arith.extf %arg0 : tensor<8x8xf32> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_extf_vector
func @test_extf_vector(%arg0 : vector<8xf32>) -> vector<8xf64> {
  %0 = arith.extf %arg0 : vector<8xf32> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_trunci
func @test_trunci(%arg0 : i32) -> i16 {
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}

// CHECK-LABEL: test_trunci_tensor
func @test_trunci_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xi16> {
  %0 = arith.trunci %arg0 : tensor<8x8xi32> to tensor<8x8xi16>
  return %0 : tensor<8x8xi16>
}

// CHECK-LABEL: test_trunci_vector
func @test_trunci_vector(%arg0 : vector<8xi32>) -> vector<8xi16> {
  %0 = arith.trunci %arg0 : vector<8xi32> to vector<8xi16>
  return %0 : vector<8xi16>
}

// CHECK-LABEL: test_truncf
func @test_truncf(%arg0 : f32) -> bf16 {
  %0 = arith.truncf %arg0 : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: test_truncf_tensor
func @test_truncf_tensor(%arg0 : tensor<8x8xf32>) -> tensor<8x8xbf16> {
  %0 = arith.truncf %arg0 : tensor<8x8xf32> to tensor<8x8xbf16>
  return %0 : tensor<8x8xbf16>
}

// CHECK-LABEL: test_truncf_vector
func @test_truncf_vector(%arg0 : vector<8xf32>) -> vector<8xbf16> {
  %0 = arith.truncf %arg0 : vector<8xf32> to vector<8xbf16>
  return %0 : vector<8xbf16>
}

// CHECK-LABEL: test_uitofp
func @test_uitofp(%arg0 : i32) -> f32 {
  %0 = arith.uitofp %arg0 : i32 to f32
 return %0 : f32
}

// CHECK-LABEL: test_uitofp_tensor
func @test_uitofp_tensor(%arg0 : tensor<8x8xi32>) -> tensor<8x8xf32> {
  %0 = arith.uitofp %arg0 : tensor<8x8xi32> to tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: test_uitofp_vector
func @test_uitofp_vector(%arg0 : vector<8xi32>) -> vector<8xf32> {
  %0 = arith.uitofp %arg0 : vector<8xi32> to vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: test_sitofp
func @test_sitofp(%arg0 : i16) -> f64 {
  %0 = arith.sitofp %arg0 : i16 to f64
  return %0 : f64
}

// CHECK-LABEL: test_sitofp_tensor
func @test_sitofp_tensor(%arg0 : tensor<8x8xi16>) -> tensor<8x8xf64> {
  %0 = arith.sitofp %arg0 : tensor<8x8xi16> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_sitofp_vector
func @test_sitofp_vector(%arg0 : vector<8xi16>) -> vector<8xf64> {
  %0 = arith.sitofp %arg0 : vector<8xi16> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_fptoui
func @test_fptoui(%arg0 : bf16) -> i8 {
  %0 = arith.fptoui %arg0 : bf16 to i8
  return %0 : i8
}

// CHECK-LABEL: test_fptoui_tensor
func @test_fptoui_tensor(%arg0 : tensor<8x8xbf16>) -> tensor<8x8xi8> {
  %0 = arith.fptoui %arg0 : tensor<8x8xbf16> to tensor<8x8xi8>
  return %0 : tensor<8x8xi8>
}

// CHECK-LABEL: test_fptoui_vector
func @test_fptoui_vector(%arg0 : vector<8xbf16>) -> vector<8xi8> {
  %0 = arith.fptoui %arg0 : vector<8xbf16> to vector<8xi8>
 return %0 : vector<8xi8>
}

// CHECK-LABEL: test_fptosi
func @test_fptosi(%arg0 : f64) -> i64 {
  %0 = arith.fptosi %arg0 : f64 to i64
  return %0 : i64
}

// CHECK-LABEL: test_fptosi_tensor
func @test_fptosi_tensor(%arg0 : tensor<8x8xf64>) -> tensor<8x8xi64> {
  %0 = arith.fptosi %arg0 : tensor<8x8xf64> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_fptosi_vector
func @test_fptosi_vector(%arg0 : vector<8xf64>) -> vector<8xi64> {
  %0 = arith.fptosi %arg0 : vector<8xf64> to vector<8xi64>
 return %0 : vector<8xi64>
}

// CHECK-LABEL: test_index_cast0
func @test_index_cast0(%arg0 : i32) -> index {
  %0 = arith.index_cast %arg0 : i32 to index
  return %0 : index
}

// CHECK-LABEL: test_index_cast_tensor0
func @test_index_cast_tensor0(%arg0 : tensor<8x8xi32>) -> tensor<8x8xindex> {
  %0 = arith.index_cast %arg0 : tensor<8x8xi32> to tensor<8x8xindex>
  return %0 : tensor<8x8xindex>
}

// CHECK-LABEL: test_index_cast_vector0
func @test_index_cast_vector0(%arg0 : vector<8xi32>) -> vector<8xindex> {
  %0 = arith.index_cast %arg0 : vector<8xi32> to vector<8xindex>
  return %0 : vector<8xindex>
}

// CHECK-LABEL: test_index_cast1
func @test_index_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_cast_tensor1
func @test_index_cast_tensor1(%arg0 : tensor<8x8xindex>) -> tensor<8x8xi64> {
  %0 = arith.index_cast %arg0 : tensor<8x8xindex> to tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: test_index_cast_vector1
func @test_index_cast_vector1(%arg0 : vector<8xindex>) -> vector<8xi64> {
  %0 = arith.index_cast %arg0 : vector<8xindex> to vector<8xi64>
  return %0 : vector<8xi64>
}

// CHECK-LABEL: test_bitcast0
func @test_bitcast0(%arg0 : i64) -> f64 {
  %0 = arith.bitcast %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: test_bitcast_tensor0
func @test_bitcast_tensor0(%arg0 : tensor<8x8xi64>) -> tensor<8x8xf64> {
  %0 = arith.bitcast %arg0 : tensor<8x8xi64> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_bitcast_vector0
func @test_bitcast_vector0(%arg0 : vector<8xi64>) -> vector<8xf64> {
  %0 = arith.bitcast %arg0 : vector<8xi64> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_bitcast1
func @test_bitcast1(%arg0 : f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: test_bitcast_tensor1
func @test_bitcast_tensor1(%arg0 : tensor<8x8xf32>) -> tensor<8x8xi32> {
  %0 = arith.bitcast %arg0 : tensor<8x8xf32> to tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: test_bitcast_vector1
func @test_bitcast_vector1(%arg0 : vector<8xf32>) -> vector<8xi32> {
  %0 = arith.bitcast %arg0 : vector<8xf32> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: test_cmpi
func @test_cmpi(%arg0 : i64, %arg1 : i64) -> i1 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  return %0 : i1
}

// CHECK-LABEL: test_cmpi_tensor
func @test_cmpi_tensor(%arg0 : tensor<8x8xi64>, %arg1 : tensor<8x8xi64>) -> tensor<8x8xi1> {
  %0 = arith.cmpi slt, %arg0, %arg1 : tensor<8x8xi64>
  return %0 : tensor<8x8xi1>
}

// CHECK-LABEL: test_cmpi_vector
func @test_cmpi_vector(%arg0 : vector<8xi64>, %arg1 : vector<8xi64>) -> vector<8xi1> {
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<8xi64>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: test_cmpf
func @test_cmpf(%arg0 : f64, %arg1 : f64) -> i1 {
  %0 = arith.cmpf oeq, %arg0, %arg1 : f64
  return %0 : i1
}

// CHECK-LABEL: test_cmpf_tensor
func @test_cmpf_tensor(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>) -> tensor<8x8xi1> {
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<8x8xf64>
  return %0 : tensor<8x8xi1>
}

// CHECK-LABEL: test_cmpf_vector
func @test_cmpf_vector(%arg0 : vector<8xf64>, %arg1 : vector<8xf64>) -> vector<8xi1> {
  %0 = arith.cmpf ult, %arg0, %arg1 : vector<8xf64>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: test_index_cast
func @test_index_cast(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_cast_tensor
func @test_index_cast_tensor(%arg0 : tensor<index>) -> tensor<i64> {
  %0 = arith.index_cast %arg0 : tensor<index> to tensor<i64>
  return %0 : tensor<i64>
}

// CHECK-LABEL: test_index_cast_tensor_reverse
func @test_index_cast_tensor_reverse(%arg0 : tensor<i64>) -> tensor<index> {
  %0 = arith.index_cast %arg0 : tensor<i64> to tensor<index>
  return %0 : tensor<index>
}

// CHECK-LABEL: func @bitcast(
func @bitcast(%arg : f32) -> i32 {
  %res = arith.bitcast %arg : f32 to i32
  return %res : i32
}

// CHECK-LABEL: test_constant
func @test_constant() -> () {
  // CHECK: %c42_i32 = arith.constant 42 : i32
  %0 = "arith.constant"(){value = 42 : i32} : () -> i32

  // CHECK: %c42_i32_0 = arith.constant 42 : i32
  %1 = arith.constant 42 : i32

  // CHECK: %c43 = arith.constant {crazy = "std.foo"} 43 : index
  %2 = arith.constant {crazy = "std.foo"} 43: index

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

  return
}
