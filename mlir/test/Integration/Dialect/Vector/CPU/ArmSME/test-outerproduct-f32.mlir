// DEFINE: %{entry_point} = test_outerproduct_4x4xf32
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -enable-arm-streaming="mode=locally enable-za" \
// DEFINE:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-sme" -cse -canonicalize \
// DEFINE:   -allocate-arm-sme-tiles -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme \
// DEFINE:   -e %{entry_point} -entry-point-result=void \
// DEFINE:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

// REDEFINE: %{entry_point} = test_outerproduct_accumulator_4x4xf32
// RUN: %{compile} | %{run} | FileCheck %s --check-prefix=CHECK-ACC

func.func @test_outerproduct_4x4xf32() {
  %c0 = arith.constant 0 : index

  %vector_i32 = llvm.intr.experimental.stepvector : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>
  %tile = vector.outerproduct %vector, %vector : vector<[4]xf32>, vector<[4]xf32>

  // Calculate the size of a 32-bit tile, e.g. ZA{n}.s.
  %vscale = vector.vscale
  %min_elts_s = arith.constant 4 : index
  %svl_s = arith.muli %min_elts_s, %vscale : index
  %za_s_size = arith.muli %svl_s, %svl_s : index

  // Allocate memory.
  %mem = memref.alloca(%za_s_size) : memref<?xf32>

  // Store the tile to memory.
  vector.store %tile, %mem[%c0] : memref<?xf32>, vector<[4]x[4]xf32>

  // Reload and print. The smallest SVL is 128-bits so the tile will be at
  // least 4x4xf32.
  //
  // CHECK:      ( 0, 0, 0, 0
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 0, 2, 4, 6
  // CHECK-NEXT: ( 0, 3, 6, 9
  scf.for %i = %c0 to %za_s_size step %svl_s {
    %tileslice = vector.load %mem[%i] : memref<?xf32>, vector<[4]xf32>
    vector.print %tileslice : vector<[4]xf32>
  }

  return
}

func.func @test_outerproduct_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index
  %f10 = arith.constant 10.0 : f32

  %acc = vector.broadcast %f10 : f32 to vector<[4]x[4]xf32>
  %vector_i32 = llvm.intr.experimental.stepvector : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>
  %tile = vector.outerproduct %vector, %vector, %acc : vector<[4]xf32>, vector<[4]xf32>

  // Calculate the size of a 32-bit tile, e.g. ZA{n}.s.
  %vscale = vector.vscale
  %min_elts_s = arith.constant 4 : index
  %svl_s = arith.muli %min_elts_s, %vscale : index
  %za_s_size = arith.muli %svl_s, %svl_s : index

  // Allocate memory.
  %mem = memref.alloca(%za_s_size) : memref<?xf32>

  // Store the tile to memory.
  vector.store %tile, %mem[%c0] : memref<?xf32>, vector<[4]x[4]xf32>

  // Reload and print. The smallest SVL is 128-bits so the tile will be at
  // least 4x4xf32.
  //
  // CHECK-ACC:      ( 10, 10, 10, 10
  // CHECK-ACC-NEXT: ( 10, 11, 12, 13
  // CHECK-ACC-NEXT: ( 10, 12, 14, 16
  // CHECK-ACC-NEXT: ( 10, 13, 16, 19
  scf.for %i = %c0 to %za_s_size step %svl_s {
    %tileslice = vector.load %mem[%i] : memref<?xf32>, vector<[4]xf32>
    vector.print %tileslice : vector<[4]xf32>
  }

  return
}
