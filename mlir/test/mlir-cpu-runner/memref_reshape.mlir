// RUN: mlir-opt %s -convert-scf-to-std -std-expand -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s


func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @main() -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // Initialize input.
  %input = memref.alloc() : memref<2x3xf32>
  %dim_x = memref.dim %input, %c0 : memref<2x3xf32>
  %dim_y = memref.dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %prod = muli %i,  %dim_y : index
    %val = addi %prod, %j : index
    %val_i64 = index_cast %val : index to i64
    %val_f32 = sitofp %val_i64 : i64 to f32
    memref.store %val_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK-NEXT: [0,   1,   2]
  // CHECK-NEXT: [3,   4,   5]

  // Initialize shape.
  %shape = memref.alloc() : memref<2xindex>
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  memref.store %c3, %shape[%c0] : memref<2xindex>
  memref.store %c2, %shape[%c1] : memref<2xindex>

  // Test cases.
  call @reshape_ranked_memref_to_ranked(%input, %shape)
    : (memref<2x3xf32>, memref<2xindex>) -> ()
  call @reshape_unranked_memref_to_ranked(%input, %shape)
    : (memref<2x3xf32>, memref<2xindex>) -> ()
  call @reshape_ranked_memref_to_unranked(%input, %shape)
    : (memref<2x3xf32>, memref<2xindex>) -> ()
  call @reshape_unranked_memref_to_unranked(%input, %shape)
    : (memref<2x3xf32>, memref<2xindex>) -> ()
  memref.dealloc %input : memref<2x3xf32>
  memref.dealloc %shape : memref<2xindex>
  return
}

func @reshape_ranked_memref_to_ranked(%input : memref<2x3xf32>,
                                      %shape : memref<2xindex>) {
  %output = memref.reshape %input(%shape)
                : (memref<2x3xf32>, memref<2xindex>) -> memref<?x?xf32>

  %unranked_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 2] strides = [2, 1] data =
  // CHECK: [0,   1],
  // CHECK: [2,   3],
  // CHECK: [4,   5]
  return
}

func @reshape_unranked_memref_to_ranked(%input : memref<2x3xf32>,
                                        %shape : memref<2xindex>) {
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  %output = memref.reshape %input(%shape)
                : (memref<2x3xf32>, memref<2xindex>) -> memref<?x?xf32>

  %unranked_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 2] strides = [2, 1] data =
  // CHECK: [0,   1],
  // CHECK: [2,   3],
  // CHECK: [4,   5]
  return
}

func @reshape_ranked_memref_to_unranked(%input : memref<2x3xf32>,
                                        %shape : memref<2xindex>) {
  %dyn_size_shape = memref.cast %shape : memref<2xindex> to memref<?xindex>
  %output = memref.reshape %input(%dyn_size_shape)
                : (memref<2x3xf32>, memref<?xindex>) -> memref<*xf32>

  call @print_memref_f32(%output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 2] strides = [2, 1] data =
  // CHECK: [0,   1],
  // CHECK: [2,   3],
  // CHECK: [4,   5]
  return
}

func @reshape_unranked_memref_to_unranked(%input : memref<2x3xf32>,
                                          %shape : memref<2xindex>) {
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  %dyn_size_shape = memref.cast %shape : memref<2xindex> to memref<?xindex>
  %output = memref.reshape %input(%dyn_size_shape)
                : (memref<2x3xf32>, memref<?xindex>) -> memref<*xf32>

  call @print_memref_f32(%output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 2] strides = [2, 1] data =
  // CHECK: [0,   1],
  // CHECK: [2,   3],
  // CHECK: [4,   5]
  return
}
