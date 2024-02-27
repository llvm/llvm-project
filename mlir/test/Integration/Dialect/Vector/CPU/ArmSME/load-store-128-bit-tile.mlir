// DEFINE: %{entry_point} = test_load_store_zaq0
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-arm-sme -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib

// RUN: %{compile} | %{run} | FileCheck %s

func.func @print_i8s(%bytes: memref<?xi8>, %len: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %len step %c16 {
    %v = vector.load %bytes[%i] : memref<?xi8>, vector<16xi8>
    vector.print %v : vector<16xi8>
  }
  return
}

func.func @vector_copy_i128(%src: memref<?x?xi128>, %dst: memref<?x?xi128>) {
  %c0 = arith.constant 0 : index
  %tile = vector.load %src[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  vector.store %tile, %dst[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

func.func @test_load_store_zaq0() {
  %c0 = arith.constant 0 : index
  %min_elts_q = arith.constant 1 : index
  %bytes_per_128_bit = arith.constant 16 : index

  /// Calculate the size of an 128-bit tile, e.g. ZA{n}.q, in bytes:
  %vscale = vector.vscale
  %svl_q = arith.muli %min_elts_q, %vscale : index
  %zaq_size = arith.muli %svl_q, %svl_q : index
  %zaq_size_bytes = arith.muli %zaq_size, %bytes_per_128_bit : index

  /// Allocate memory for two 128-bit tiles (A and B) and fill them a constant.
  /// The tiles are allocated as bytes so we can fill and print them, as there's
  /// very little that can be done with 128-bit types directly.
  %tile_a_bytes = memref.alloca(%zaq_size_bytes) {alignment = 16} : memref<?xi8>
  %tile_b_bytes = memref.alloca(%zaq_size_bytes) {alignment = 16} : memref<?xi8>
  %fill_a_i8 = arith.constant 7 : i8
  %fill_b_i8 = arith.constant 64 : i8
  linalg.fill ins(%fill_a_i8 : i8) outs(%tile_a_bytes : memref<?xi8>)
  linalg.fill ins(%fill_b_i8 : i8) outs(%tile_b_bytes : memref<?xi8>)

  /// Get an 128-bit view of the memory for tiles A and B:
  %tile_a = memref.view %tile_a_bytes[%c0][%svl_q, %svl_q] :
    memref<?xi8> to memref<?x?xi128>
  %tile_b = memref.view %tile_b_bytes[%c0][%svl_q, %svl_q] :
    memref<?xi8> to memref<?x?xi128>

  // CHECK-LABEL: INITIAL TILE A:
  // CHECK: ( 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 )
  vector.print str "INITIAL TILE A:"
  func.call @print_i8s(%tile_a_bytes, %zaq_size_bytes) : (memref<?xi8>, index) -> ()
  vector.print punctuation <newline>

  // CHECK-LABEL: INITIAL TILE B:
  // CHECK: ( 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64 )
  vector.print str "INITIAL TILE B:"
  func.call @print_i8s(%tile_b_bytes, %zaq_size_bytes) : (memref<?xi8>, index) -> ()
  vector.print punctuation <newline>

  /// Load tile A and store it to tile B:
  func.call @vector_copy_i128(%tile_a, %tile_b) : (memref<?x?xi128>, memref<?x?xi128>) -> ()

  // CHECK-LABEL: FINAL TILE A:
  // CHECK: ( 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 )
  vector.print str "FINAL TILE A:"
  func.call @print_i8s(%tile_a_bytes, %zaq_size_bytes) : (memref<?xi8>, index) -> ()
  vector.print punctuation <newline>

  // CHECK-LABEL: FINAL TILE B:
  // CHECK: ( 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 )
  vector.print str "FINAL TILE B:"
  func.call @print_i8s(%tile_b_bytes, %zaq_size_bytes) : (memref<?xi8>, index) -> ()

  return
}
