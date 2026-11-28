// RUN: mlir-opt %s --arith-expand="include-min-max-f=true" \
// RUN:   --convert-elementwise-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --canonicalize --buffer-deallocation-pipeline --convert-bufferization-to-memref \
// RUN:   --convert-linalg-to-loops --convert-scf-to-cf \
// RUN:   --convert-vector-to-llvm --convert-math-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils | FileCheck %s --match-full-lines

// Print each maximum and minimum as bits to distinguish the two zeros.
func.func @print_bits(%max: tensor<?xf32>, %min: tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %size = tensor.dim %max, %c0 : tensor<?xf32>
  scf.for %i = %c0 to %size step %c1 {
    %max_elem = tensor.extract %max[%i] : tensor<?xf32>
    %min_elem = tensor.extract %min[%i] : tensor<?xf32>
    %max_bits = arith.bitcast %max_elem : f32 to i32
    %min_bits = arith.bitcast %min_elem : f32 to i32
    vector.print %max_bits : i32
    vector.print %min_bits : i32
  }
  return
}

func.func @check_static(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) {
  %max = arith.maximumf %lhs, %rhs : tensor<4xf32>
  %min = arith.minimumf %lhs, %rhs : tensor<4xf32>
  %max_dynamic = tensor.cast %max : tensor<4xf32> to tensor<?xf32>
  %min_dynamic = tensor.cast %min : tensor<4xf32> to tensor<?xf32>
  func.call @print_bits(%max_dynamic, %min_dynamic) : (tensor<?xf32>, tensor<?xf32>) -> ()
  return
}

func.func @check_dynamic(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) {
  %max = arith.maximumf %lhs, %rhs : tensor<?xf32>
  %min = arith.minimumf %lhs, %rhs : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %max_size = tensor.dim %max, %c0 : tensor<?xf32>
  %min_size = tensor.dim %min, %c0 : tensor<?xf32>
  vector.print %max_size : index
  vector.print %min_size : index
  func.call @print_bits(%max, %min) : (tensor<?xf32>, tensor<?xf32>) -> ()
  return
}

func.func @entry() {
  %lhs = arith.constant dense<[0.0, -0.0, 0.0, -0.0]> : tensor<4xf32>
  %rhs = arith.constant dense<[-0.0, 0.0, 0.0, -0.0]> : tensor<4xf32>
  // CHECK: 0
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: -2147483648
  func.call @check_static(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> ()

  %lhs_dynamic = tensor.cast %lhs : tensor<4xf32> to tensor<?xf32>
  %rhs_dynamic = tensor.cast %rhs : tensor<4xf32> to tensor<?xf32>
  // CHECK-NEXT: 4
  // CHECK-NEXT: 4
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: -2147483648
  func.call @check_dynamic(%lhs_dynamic, %rhs_dynamic) : (tensor<?xf32>, tensor<?xf32>) -> ()

  // Run the same dynamic calculation with a different extent.
  %short_lhs = arith.constant dense<[2.0, -0.0]> : tensor<2xf32>
  %short_rhs = arith.constant dense<[-2.0, 0.0]> : tensor<2xf32>
  %short_lhs_dynamic = tensor.cast %short_lhs : tensor<2xf32> to tensor<?xf32>
  %short_rhs_dynamic = tensor.cast %short_rhs : tensor<2xf32> to tensor<?xf32>
  // CHECK-NEXT: 2
  // CHECK-NEXT: 2
  // CHECK-NEXT: 1073741824
  // CHECK-NEXT: -1073741824
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  func.call @check_dynamic(%short_lhs_dynamic, %short_rhs_dynamic) : (tensor<?xf32>, tensor<?xf32>) -> ()

  // An empty result prints only the two runtime extents.
  %empty = arith.constant dense<> : tensor<0xf32>
  %empty_dynamic = tensor.cast %empty : tensor<0xf32> to tensor<?xf32>
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NOT: {{.}}
  func.call @check_dynamic(%empty_dynamic, %empty_dynamic) : (tensor<?xf32>, tensor<?xf32>) -> ()
  return
}
