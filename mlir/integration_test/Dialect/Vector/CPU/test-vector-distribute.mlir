// RUN: mlir-opt %s -test-vector-distribute-patterns=distribution-multiplicity=32 \
// RUN:  -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

func @print_memref_f32(memref<*xf32>)

func @alloc_1d_filled_inc_f32(%arg0: index, %arg1: f32) -> memref<?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = alloc(%arg0) : memref<?xf32>
  scf.for %arg2 = %c0 to %arg0 step %c1 {
    %tmp = index_cast %arg2 : index to i32
    %tmp1 = sitofp %tmp : i32 to f32
    %tmp2 = addf %tmp1, %arg1 : f32
    store %tmp2, %0[%arg2] : memref<?xf32>
  }
  return %0 : memref<?xf32>
}

func @vector_add_cycle(%id : index, %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0], %cf0: memref<?xf32>, vector<64xf32>
  %b = vector.transfer_read %B[%c0], %cf0: memref<?xf32>, vector<64xf32>
  %acc = addf %a, %b: vector<64xf32>
  vector.transfer_write %acc, %C[%c0]: vector<64xf32>, memref<?xf32>
  return
}

// Loop over a function containinng a large add vector and distribute it so that
// each iteration of the loop process part of the vector operation.
func @main() {
  %cf1 = constant 1.0 : f32
  %cf2 = constant 2.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c64 = constant 64 : index
  %out = alloc(%c64) : memref<?xf32>
  %in1 = call @alloc_1d_filled_inc_f32(%c64, %cf1) : (index, f32) -> memref<?xf32>
  %in2 = call @alloc_1d_filled_inc_f32(%c64, %cf2) : (index, f32) -> memref<?xf32>
  scf.for %arg5 = %c0 to %c64 step %c1 {
    call @vector_add_cycle(%arg5, %in1, %in2, %out) : (index, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  }
  %converted = memref_cast %out : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%converted): (memref<*xf32>) -> ()
  // CHECK:      Unranked{{.*}}data =
  // CHECK:      [
  // CHECK-SAME:  3,  5,  7,  9,  11,  13,  15,  17,  19,  21,  23,  25,  27,
  // CHECK-SAME:  29,  31,  33,  35,  37,  39,  41,  43,  45,  47,  49,  51,
  // CHECK-SAME:  53,  55,  57,  59,  61,  63,  65,  67,  69,  71,  73,  75,
  // CHECK-SAME:  77,  79,  81,  83,  85,  87,  89,  91,  93,  95,  97,  99,
  // CHECK-SAME:  101,  103,  105,  107,  109,  111,  113,  115,  117,  119,
  // CHECK-SAME:  121,  123,  125,  127,  129]
  dealloc %out : memref<?xf32>
  dealloc %in1 : memref<?xf32>
  dealloc %in2 : memref<?xf32>
  return
}
