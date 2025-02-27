// DEFINE: %{entry_point} = test_outerproduct_no_accumulator_2x2xf64
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -test-lower-to-arm-sme -test-lower-to-llvm -o %t
// DEFINE: %{run} = %mcr_aarch64_cmd %t \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme-f64f64 \
// DEFINE:   -e %{entry_point} -entry-point-result=void \
// DEFINE:   -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib

// RUN: %{compile}

// RUN: %{run} | FileCheck %s

// REDEFINE: %{entry_point} = test_outerproduct_with_accumulator_2x2xf64
// RUN: %{run} | FileCheck %s --check-prefix=WITH-ACC

// REDEFINE: %{entry_point} = test_masked_outerproduct_no_accumulator_2x2xf64
// RUN: %{run} | FileCheck %s --check-prefix=WITH-MASK

// REDEFINE: %{entry_point} = test_masked_outerproduct_with_accumulator_2x2xf64
// RUN: %{run} | FileCheck %s --check-prefix=WITH-MASK-AND-ACC

func.func @test_outerproduct_no_accumulator_2x2xf64() {
  %c0 = arith.constant 0 : index
  %ones = arith.constant dense<1> : vector<[2]xi32>

  %step_vector = llvm.intr.stepvector : vector<[2]xi32>
  %vector_i32 = arith.addi %step_vector, %ones : vector<[2]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[2]xi32> to vector<[2]xf64>

  %lhsDim = arith.constant 1 : index
  %rhsDim = arith.constant 2 : index
  %mask = vector.create_mask %lhsDim, %rhsDim : vector<[2]x[2]xi1>

  %tile = vector.outerproduct %vector, %vector : vector<[2]xf64>, vector<[2]xf64>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 2x2xf64.
  //
  // CHECK:      TILE BEGIN
  // CHECK-NEXT: ( 1, 2
  // CHECK-NEXT: ( 2, 4
  // CHECK:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[2]x[2]xf64>
  vector.print str "TILE END\n"

  return
}

func.func @test_outerproduct_with_accumulator_2x2xf64() {
  %c0 = arith.constant 0 : index
  %ones = arith.constant dense<1> : vector<[2]xi32>
  %f10 = arith.constant 10.0 : f64

  %acc = vector.splat %f10 : vector<[2]x[2]xf64>
  %step_vector = llvm.intr.stepvector : vector<[2]xi32>
  %vector_i32 = arith.addi %step_vector, %ones : vector<[2]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[2]xi32> to vector<[2]xf64>

  %tile = vector.outerproduct %vector, %vector, %acc : vector<[2]xf64>, vector<[2]xf64>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 2x2xf64.
  //
  // WITH-ACC:      TILE BEGIN
  // WITH-ACC-NEXT: ( 11, 12
  // WITH-ACC-NEXT: ( 12, 14
  // WITH-ACC:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[2]x[2]xf64>
  vector.print str "TILE END\n"

  return
}

func.func @test_masked_outerproduct_no_accumulator_2x2xf64() {
  %c0 = arith.constant 0 : index
  %ones = arith.constant dense<1> : vector<[2]xi32>
  %f10 = arith.constant 10.0 : f64

  %step_vector = llvm.intr.stepvector : vector<[2]xi32>
  %vector_i32 = arith.addi %step_vector, %ones : vector<[2]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[2]xi32> to vector<[2]xf64>

  %lhsDim = arith.constant 2 : index
  %rhsDim = arith.constant 1 : index
  %mask = vector.create_mask %lhsDim, %rhsDim : vector<[2]x[2]xi1>

  %tile = vector.mask %mask {
    vector.outerproduct %vector, %vector : vector<[2]xf64>, vector<[2]xf64>
  } : vector<[2]x[2]xi1> -> vector<[2]x[2]xf64>

  // Print the tile. Due to masking the result will be the top 2x1xf64 section.
  //
  // WITH-MASK:      TILE BEGIN
  // WITH-MASK-NEXT: ( 1, 0
  // WITH-MASK-NEXT: ( 2, 0
  // WITH-MASK:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[2]x[2]xf64>
  vector.print str "TILE END\n"

  return
}

func.func @test_masked_outerproduct_with_accumulator_2x2xf64() {
  %c0 = arith.constant 0 : index
  %ones = arith.constant dense<1> : vector<[2]xi32>
  %f10 = arith.constant 10.0 : f64

  %acc = vector.splat %f10 : vector<[2]x[2]xf64>
  %step_vector = llvm.intr.stepvector : vector<[2]xi32>
  %vector_i32 = arith.addi %step_vector, %ones : vector<[2]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[2]xi32> to vector<[2]xf64>

  %lhsDim = arith.constant 1 : index
  %rhsDim = arith.constant 2 : index
  %mask = vector.create_mask %lhsDim, %rhsDim : vector<[2]x[2]xi1>

  %tile = vector.mask %mask {
    vector.outerproduct %vector, %vector, %acc : vector<[2]xf64>, vector<[2]xf64>
  } : vector<[2]x[2]xi1> -> vector<[2]x[2]xf64>

  // Print the tile. Due to masking the result will be the top 1x2xf64 section.
  //
  // WITH-MASK-AND-ACC:      TILE BEGIN
  // WITH-MASK-AND-ACC-NEXT: ( 11, 12
  // WITH-MASK-AND-ACC-NEXT: ( 10, 10
  // WITH-MASK-AND-ACC:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %tile : vector<[2]x[2]xf64>
  vector.print str "TILE END\n"

  return
}
