// RUN: mlir-opt %s -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format allow-pattern-rollback=0" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// A rank-zero memref registers one element, even though its descriptor has no
// size or stride entries.
func.func @main() {
  %buffer = memref.alloc() : memref<f32>
  %unranked = memref.cast %buffer : memref<f32> to memref<*xf32>
  %pointer = memref.extract_aligned_pointer_as_index %buffer : memref<f32> -> index
  %address = arith.index_cast %pointer : index to i64
  %memoryType = memref.alloca() : memref<i32>
  // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
  %attribute = arith.constant 2 : i32

  gpu.host_register %unranked : memref<*xf32>
  %registered = call @cuPointerGetAttribute(%memoryType, %attribute, %address)
      : (memref<i32>, i32, i64) -> i32
  // CUDA_SUCCESS: the pointer belongs to registered host memory.
  // CHECK: 0
  vector.print %registered : i32

  gpu.host_unregister %unranked : memref<*xf32>
  %unregistered = call @cuPointerGetAttribute(%memoryType, %attribute, %address)
      : (memref<i32>, i32, i64) -> i32
  // CUDA_ERROR_INVALID_VALUE: the pointer is no longer registered.
  // CHECK-NEXT: 1
  vector.print %unregistered : i32

  memref.dealloc %buffer : memref<f32>
  return
}

func.func private @cuPointerGetAttribute(memref<i32>, i32, i64) -> i32
    attributes {llvm.bareptr}
