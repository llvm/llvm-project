// RUN: mlir-opt %s -test-vector-to-forloop -convert-vector-to-scf \
// RUN:   -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine \
// RUN: -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | mlir-cpu-runner -e main \
// RUN: -entry-point-result=void \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -test-vector-to-forloop | FileCheck %s -check-prefix=TRANSFORM


func private @print_memref_f32(memref<*xf32>)

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

// Large vector addf that can be broken down into a loop of smaller vector addf.
func @main() {
  %cf0 = constant 0.0 : f32
  %cf1 = constant 1.0 : f32
  %cf2 = constant 2.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c32 = constant 32 : index
  %c64 = constant 64 : index
  %out = alloc(%c64) : memref<?xf32>
  %in1 = call @alloc_1d_filled_inc_f32(%c64, %cf1) : (index, f32) -> memref<?xf32>
  %in2 = call @alloc_1d_filled_inc_f32(%c64, %cf2) : (index, f32) -> memref<?xf32>
  // Check that the tansformatio correctly happened.
  // TRANSFORM: scf.for
  // TRANSFORM:   vector.transfer_read {{.*}} : memref<?xf32>, vector<2xf32>
  // TRANSFORM:   vector.transfer_read {{.*}} : memref<?xf32>, vector<2xf32>
  // TRANSFORM:   %{{.*}} = addf %{{.*}}, %{{.*}} : vector<2xf32>
  // TRANSFORM:   vector.transfer_write {{.*}} : vector<2xf32>, memref<?xf32>
  // TRANSFORM: }
  %a = vector.transfer_read %in1[%c0], %cf0: memref<?xf32>, vector<64xf32>
  %b = vector.transfer_read %in2[%c0], %cf0: memref<?xf32>, vector<64xf32>
  %acc = addf %a, %b: vector<64xf32>
  vector.transfer_write %acc, %out[%c0]: vector<64xf32>, memref<?xf32>
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
