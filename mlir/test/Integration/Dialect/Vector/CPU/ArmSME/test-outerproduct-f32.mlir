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

// REDEFINE: %{entry_point} = test_outerproduct_no_accumulator_4x4xf32
// RUN: %{compile} | %{run} | FileCheck %s --check-prefix=CHECK-NO-ACC

func.func @test_outerproduct_4x4xf32() {
  %c0 = arith.constant 0 : index
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f10 = arith.constant 10.0 : f32

  %a = vector.splat %f1 : vector<[4]xf32>
  %b = vector.splat %f2 : vector<[4]xf32>
  // TODO: vector.splat doesn't support ArmSME.
  %c = vector.broadcast %f10 : f32 to vector<[4]x[4]xf32>

  %tile = vector.outerproduct %a, %b, %c : vector<[4]xf32>, vector<[4]xf32>

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
  // CHECK:      ( 12, 12, 12, 12
  // CHECK-NEXT: ( 12, 12, 12, 12
  // CHECK-NEXT: ( 12, 12, 12, 12
  // CHECK-NEXT: ( 12, 12, 12, 12
  scf.for %i = %c0 to %za_s_size step %svl_s {
    %tileslice = vector.load %mem[%i] : memref<?xf32>, vector<[4]xf32>
    vector.print %tileslice : vector<[4]xf32>
  }

  return
}

func.func @test_outerproduct_no_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f10 = arith.constant 10.0 : f32

  %a = vector.splat %f1 : vector<[4]xf32>
  %b = vector.splat %f2 : vector<[4]xf32>

  %tile = vector.outerproduct %a, %b : vector<[4]xf32>, vector<[4]xf32>

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
  // CHECK-NO-ACC:      ( 2, 2, 2, 2
  // CHECK-NO-ACC-NEXT: ( 2, 2, 2, 2
  // CHECK-NO-ACC-NEXT: ( 2, 2, 2, 2
  // CHECK-NO-ACC-NEXT: ( 2, 2, 2, 2
  scf.for %i = %c0 to %za_s_size step %svl_s {
    %tileslice = vector.load %mem[%i] : memref<?xf32>, vector<[4]xf32>
    vector.print %tileslice : vector<[4]xf32>
  }

  return
}
