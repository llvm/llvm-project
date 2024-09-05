// RUN: mlir-opt %s -convert-vector-to-scf -arm-sve-legalize-vector-storage -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd -e=entry -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%native_mlir_c_runner_utils | \
// RUN: FileCheck %s

/// This tests basic functionality of arrays of scalable vectors, which in MLIR
/// are vectors with a single trailing scalable dimension. This test requires
/// the -arm-sve-legalize-vector-storage pass to ensure the loads/stores done
/// here are be legal for the LLVM backend.

func.func @read_and_print_2d_vector(%memref: memref<3x?xf32>)  {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim = memref.dim %memref, %c1 : memref<3x?xf32>
  %mask = vector.create_mask %c2, %dim : vector<3x[8]xi1>
  %vector = vector.transfer_read %memref[%c0,%c0], %cst, %mask {in_bounds = [true, true]} : memref<3x?xf32>, vector<3x[8]xf32>

  /// TODO: Support vector.print for arrays of scalable vectors.
  %row0 = vector.extract %vector[0] : vector<[8]xf32> from vector<3x[8]xf32>
  %row1 = vector.extract %vector[1] : vector<[8]xf32> from vector<3x[8]xf32>
  %row2 = vector.extract %vector[2] : vector<[8]xf32> from vector<3x[8]xf32>

  /// Print each of the vectors.
  /// vscale is >= 1, so at least 8 elements will be printed.

  vector.print str "read_and_print_2d_vector()\n"
  // CHECK-LABEL: read_and_print_2d_vector()
  // CHECK: ( 8, 8, 8, 8, 8, 8, 8, 8
  vector.print %row0 : vector<[8]xf32>
  // CHECK: ( 8, 8, 8, 8, 8, 8, 8, 8
  vector.print %row1 : vector<[8]xf32>
  /// This last row is all zero due to our mask.
  // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0
  vector.print %row2 : vector<[8]xf32>

  return
}

func.func @print_1x2xVSCALExf32(%vector: vector<1x2x[4]xf32>) {
  /// TODO: Support vector.print for arrays of scalable vectors.
  %slice0 = vector.extract %vector[0, 1] : vector<[4]xf32> from vector<1x2x[4]xf32>
  %slice1 = vector.extract %vector[0, 1] : vector<[4]xf32> from vector<1x2x[4]xf32>
  vector.print %slice0 : vector<[4]xf32>
  vector.print %slice1 : vector<[4]xf32>
  return
}

func.func @add_arrays_of_scalable_vectors(%a: memref<1x2x?xf32>, %b: memref<1x2x?xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim_a = memref.dim %a, %c2 : memref<1x2x?xf32>
  %dim_b = memref.dim %b, %c2 : memref<1x2x?xf32>
  %mask_a = vector.create_mask %c2, %c3, %dim_a : vector<1x2x[4]xi1>
  %mask_b = vector.create_mask %c2, %c3, %dim_b : vector<1x2x[4]xi1>

  /// Print each of the vectors.
  /// vscale is >= 1, so at least 4 elements will be printed.

  // CHECK-LABEL: Vector A
  // CHECK-NEXT: ( 5, 5, 5, 5
  // CHECK-NEXT: ( 5, 5, 5, 5
  vector.print str "\nVector A\n"
  %vector_a = vector.transfer_read %a[%c0, %c0, %c0], %cst, %mask_a {in_bounds = [true, true, true]} : memref<1x2x?xf32>, vector<1x2x[4]xf32>
  func.call @print_1x2xVSCALExf32(%vector_a) : (vector<1x2x[4]xf32>) -> ()

  // CHECK-LABEL: Vector B
  // CHECK-NEXT: ( 4, 4, 4, 4
  // CHECK-NEXT: ( 4, 4, 4, 4
  vector.print str "\nVector B\n"
  %vector_b = vector.transfer_read %b[%c0, %c0, %c0], %cst, %mask_b {in_bounds = [true, true, true]} : memref<1x2x?xf32>, vector<1x2x[4]xf32>
  func.call @print_1x2xVSCALExf32(%vector_b) : (vector<1x2x[4]xf32>) -> ()

  // CHECK-LABEL: Sum
  // CHECK-NEXT: ( 9, 9, 9, 9
  // CHECK-NEXT: ( 9, 9, 9, 9
  vector.print str "\nSum\n"
  %sum = arith.addf %vector_a, %vector_b : vector<1x2x[4]xf32>
  func.call @print_1x2xVSCALExf32(%sum) : (vector<1x2x[4]xf32>) -> ()

  return
}

func.func @entry() {
  %vscale = vector.vscale

  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %f32_8 = arith.constant 8.0 : f32
  %f32_5 = arith.constant 5.0 : f32
  %f32_4 = arith.constant 4.0 : f32

  %test_1_memref_size = arith.muli %vscale, %c8 : index
  %test_1_memref = memref.alloca(%test_1_memref_size) : memref<3x?xf32>

  linalg.fill ins(%f32_8 : f32) outs(%test_1_memref :memref<3x?xf32>)

  vector.print str "=> Print and read 2D arrays of scalable vectors:\n"
  func.call @read_and_print_2d_vector(%test_1_memref) : (memref<3x?xf32>) -> ()

  vector.print str "\n====================\n"

  %test_2_memref_size = arith.muli %vscale, %c4 : index
  %test_2_memref_a = memref.alloca(%test_2_memref_size) : memref<1x2x?xf32>
  %test_2_memref_b = memref.alloca(%test_2_memref_size) : memref<1x2x?xf32>

  linalg.fill ins(%f32_5 : f32) outs(%test_2_memref_a :memref<1x2x?xf32>)
  linalg.fill ins(%f32_4 : f32) outs(%test_2_memref_b :memref<1x2x?xf32>)

  vector.print str "=> Reading and adding two 3D arrays of scalable vectors:\n"
  func.call @add_arrays_of_scalable_vectors(
    %test_2_memref_a, %test_2_memref_b) : (memref<1x2x?xf32>, memref<1x2x?xf32>) -> ()

  return
}
