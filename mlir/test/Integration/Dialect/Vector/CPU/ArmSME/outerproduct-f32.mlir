// DEFINE: %{entry_point} = test_outerproduct_no_accumulator_4x4xf32
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -test-lower-to-arm-sme -test-lower-to-llvm -o %t
// DEFINE: %{run} = %mcr_aarch64_cmd %t \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme \
// DEFINE:   -e %{entry_point} -entry-point-result=void \
// DEFINE:   -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib

// RUN: %{compile}

// RUN: %{run} | FileCheck %s --check-prefix=WITHOUT-ACC

// REDEFINE: %{entry_point} = test_outerproduct_with_accumulator_4x4xf32
// RUN: %{run} | FileCheck %s --check-prefix=WITH-ACC

// REDEFINE: %{entry_point} = test_masked_outerproduct_no_accumulator_4x4xf32
// RUN: %{run} | FileCheck %s --check-prefix=WITH-MASK

// REDEFINE: %{entry_point} = test_masked_outerproduct_with_accumulator_4x4xf32
// RUN: %{run} | FileCheck %s --check-prefix=WITH-MASK-AND-ACC

func.func @test_outerproduct_no_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index

  %vector_i32 = llvm.intr.stepvector : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>
  %tile = vector.outerproduct %vector, %vector : vector<[4]xf32>, vector<[4]xf32>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 4x4xf32.
  //
  // WITHOUT-ACC:      TILE BEGIN
  // WITHOUT-ACC-NEXT: ( 0, 0, 0, 0
  // WITHOUT-ACC-NEXT: ( 0, 1, 2, 3
  // WITHOUT-ACC-NEXT: ( 0, 2, 4, 6
  // WITHOUT-ACC-NEXT: ( 0, 3, 6, 9
  // WITHOUT-ACC:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[4]x[4]xf32>
  vector.print str "TILE END\n"

  return
}

func.func @test_outerproduct_with_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index
  %f10 = arith.constant 10.0 : f32

  %acc = vector.broadcast %f10 : f32 to vector<[4]x[4]xf32>
  %vector_i32 = llvm.intr.stepvector : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>
  %tile = vector.outerproduct %vector, %vector, %acc : vector<[4]xf32>, vector<[4]xf32>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 4x4xf32.
  //
  // WITH-ACC:      TILE BEGIN
  // WITH-ACC-NEXT: ( 10, 10, 10, 10
  // WITH-ACC-NEXT: ( 10, 11, 12, 13
  // WITH-ACC-NEXT: ( 10, 12, 14, 16
  // WITH-ACC-NEXT: ( 10, 13, 16, 19
  // WITH-ACC:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[4]x[4]xf32>
  vector.print str "TILE END\n"

  return
}

func.func @test_masked_outerproduct_no_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index
  %ones = arith.constant dense<1> : vector<[4]xi32>

  %step_vector = llvm.intr.stepvector : vector<[4]xi32>
  %vector_i32 = arith.addi %step_vector, %ones : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>

  %lhsDim = arith.constant 3 : index
  %rhsDim = arith.constant 2 : index
  %mask = vector.create_mask %lhsDim, %rhsDim : vector<[4]x[4]xi1>

  %tile = vector.mask %mask {
    vector.outerproduct %vector, %vector : vector<[4]xf32>, vector<[4]xf32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>

  // Print the tile. Due to masking the result will be the top 3x2xf32 section.
  //
  // WITH-MASK:      TILE BEGIN
  // WITH-MASK-NEXT: ( 1, 2, 0, 0
  // WITH-MASK-NEXT: ( 2, 4, 0, 0
  // WITH-MASK-NEXT: ( 3, 6, 0, 0
  // WITH-MASK-NEXT: ( 0, 0, 0, 0
  // WITH-MASK:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[4]x[4]xf32>
  vector.print str "TILE END\n"

  return
}

func.func @test_masked_outerproduct_with_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index
  %ones = arith.constant dense<1> : vector<[4]xi32>
  %f10 = arith.constant 10.0 : f32

  %acc = vector.broadcast %f10 : f32 to vector<[4]x[4]xf32>
  %step_vector = llvm.intr.stepvector : vector<[4]xi32>
  %vector_i32 = arith.addi %step_vector, %ones : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>

  %lhsDim = arith.constant 2 : index
  %rhsDim = arith.constant 3 : index
  %mask = vector.create_mask %lhsDim, %rhsDim : vector<[4]x[4]xi1>

  %tile = vector.mask %mask {
    vector.outerproduct %vector, %vector, %acc : vector<[4]xf32>, vector<[4]xf32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>

  // Print the tile. Due to masking the result will be the top 2x3xf32 section.
  //
  // WITH-MASK-AND-ACC:      TILE BEGIN
  // WITH-MASK-AND-ACC-NEXT: ( 11, 12, 13, 10
  // WITH-MASK-AND-ACC-NEXT: ( 12, 14, 16, 10
  // WITH-MASK-AND-ACC-NEXT: ( 10, 10, 10, 10
  // WITH-MASK-AND-ACC-NEXT: ( 10, 10, 10, 10
  // WITH-MASK-AND-ACC:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[4]x[4]xf32>
  vector.print str "TILE END\n"

  return
}
