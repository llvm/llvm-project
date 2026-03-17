// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner -e entry -entry-point-result=void \
// DEFINE:         -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

/// TEST 1. Verify default compilation (direct lowering of `vector.gather` to LLVM)
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-llvm
// RUN: %{compile} | %{run} | FileCheck %s

/// TEST 2. Verify compilation via `test-vector-gather-lowering` (`vector.gather`
/// lowerd to LLVM via `vector.load`)
// REDEFINE: %{compile} = mlir-opt %s --test-vector-gather-lowering | mlir-opt -test-lower-to-llvm
// RUN: %{compile} | %{run} | FileCheck %s

/// TEST 3. Verify that `test-vector-gather-lowering` will indeed produce
/// `vector.load`
// REDEFINE: %{compile} = mlir-opt %s --test-vector-gather-lowering
// RUN: %{compile} | FileCheck %s -check-prefix CHECK-IR 

func.func @gather8(%base: memref<?x?xf32>, %indices: vector<8xi32>,
              %mask: vector<8xi1>, %pass_thru: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0: index
  /// Verify that the lowering via vector.load does indeed generate vector.load
  // CHECK-IR-COUNT-4: vector.load
  %g = vector.gather %base[%c0, %c0][%indices], %mask, %pass_thru
    : memref<?x?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  return %g : vector<8xf32>
}

func.func @entry() {
  // Set up memory.
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c10 = arith.constant 10: index
  %c5 = arith.constant 5: index
  %A = memref.alloc(%c10, %c5) : memref<?x?xf32>
  scf.for %i = %c0 to %c10 step %c1 {
    scf.for %j = %c0 to %c5 step %c1 {
      %off = arith.muli %i, %c10 : index
      %val_index = arith.addi %j, %off : index
      %val_i32 = arith.index_cast %val_index : index to i32
      %val = arith.sitofp %val_i32 : i32 to f32
      memref.store %val, %A[%i, %j] : memref<?x?xf32>
    }
  }
  %A_cast = memref.cast %A : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()

  // Set up idx vector.
  %i0 = arith.constant 0: i32
  %0 = vector.broadcast %i0 : i32 to vector<8xi32>
  %i6 = arith.constant 16: i32
  %1 = vector.insert %i6, %0[1] : i32 into vector<8xi32>
  %i1 = arith.constant 11: i32
  %2 = vector.insert %i1, %1[2] : i32 into vector<8xi32>
  %i3 = arith.constant 33: i32
  %3 = vector.insert %i3, %2[3] : i32 into vector<8xi32>
  %i5 = arith.constant 5: i32
  %4 = vector.insert %i5, %3[4] : i32 into vector<8xi32>
  %i4 = arith.constant 44: i32
  %5 = vector.insert %i4, %4[5] : i32 into vector<8xi32>
  %i9 = arith.constant 19: i32
  %6 = vector.insert %i9, %5[6] : i32 into vector<8xi32>
  %i2 = arith.constant 22: i32
  %idx = vector.insert %i2, %6[7] : i32 into vector<8xi32>

  // Set up pass thru vector.
  %u = arith.constant -7.0: f32
  %pass = vector.broadcast %u : f32 to vector<8xf32>

  // Set up masks.
  %t = arith.constant 1: i1
  %none = vector.constant_mask [0] : vector<8xi1>
  %all = vector.constant_mask [8] : vector<8xi1>
  %some = vector.constant_mask [4] : vector<8xi1>
  %more = vector.insert %t, %some[7] : i1 into vector<8xi1>

  //
  // Gather tests.
  //

  %g1 = call @gather8(%A, %idx, %all, %pass)
    : (memref<?x?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>)
    -> (vector<8xf32>)
  vector.print %g1 : vector<8xf32>
  // CHECK: ( 0, 31, 21, 63, 10, 84, 34, 42 )

  %g2 = call @gather8(%A, %idx, %none, %pass)
    : (memref<?x?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>)
    -> (vector<8xf32>)
  vector.print %g2 : vector<8xf32>
  // CHECK: ( -7, -7, -7, -7, -7, -7, -7, -7 )

  %g3 = call @gather8(%A, %idx, %some, %pass)
    : (memref<?x?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>)
    -> (vector<8xf32>)
  vector.print %g3 : vector<8xf32>
  // CHECK: ( 0, 31, 21, 63, -7, -7, -7, -7 )

  %g4 = call @gather8(%A, %idx, %more, %pass)
    : (memref<?x?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>)
    -> (vector<8xf32>)
  vector.print %g4 : vector<8xf32>
  // CHECK: ( 0, 31, 21, 63, -7, -7, -7, 42 )

  memref.dealloc %A : memref<?x?xf32>
  return
}
func.func private @printMemrefF32(%ptr : memref<*xf32>)
