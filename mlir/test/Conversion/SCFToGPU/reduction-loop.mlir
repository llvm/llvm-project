// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-affine-for-to-gpu{gpu-block-dims=1 gpu-thread-dims=1}))" %s | FileCheck %s

/// Test parallelization legality checks in affine-for-to-gpu conversion.
/// The pass is configured to map the first 2 loops to GPU block (depth 0) and
/// GPU thread (depth 1) respectively.

// CHECK-LABEL: func @map_to_gpu_inner_dep_unmapped
// CHECK-SAME: %[[MEM:.*]]: memref<10x10x10xf32>
func.func @map_to_gpu_inner_dep_unmapped(%mem: memref<10x10x10xf32>) {
  /// The inner loop 'k' (depth=2) carries a dependency. However, since the
  /// mapping only covers depth 0 and 1, 'k' remains sequential inside the
  /// GPU kernel. The outer loops are dependency-free and safe to map.

  // CHECK: gpu.launch
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      // CHECK: affine.for %{{.*}} = 1 to 10
      // CHECK: affine.load %[[MEM]]
      // CHECK: affine.store %{{.*}}, %[[MEM]]
      affine.for %k = 1 to 10 {
         %0 = affine.load %mem[%i, %j, %k - 1] : memref<10x10x10xf32>
         affine.store %0, %mem[%i, %j, %k] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: func @negative_map_to_gpu_block_dep
func.func @negative_map_to_gpu_block_dep(%mem: memref<10xf32>) {
  /// The loop 'i' is mapped to a block dimension (depth=0).
  /// The loop-carried dependency makes parallelization unsafe.

  // CHECK-NOT: gpu.launch
  // CHECK: affine.for
  affine.for %i = 1 to 10 {
     %0 = affine.load %mem[%i - 1] : memref<10xf32>
     affine.store %0, %mem[%i] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func @negative_map_to_gpu_thread_dep
func.func @negative_map_to_gpu_thread_dep(%mem: memref<10x10xf32>) {
  /// The inner loop 'j' is mapped to a thread dimension (depth=1).
  /// A dependency in any mapped loop invalidates the entire nest conversion.

  // CHECK-NOT: gpu.launch
  // CHECK: affine.for
  affine.for %i = 0 to 10 {
    // CHECK: affine.for
    affine.for %j = 1 to 10 {
       %0 = affine.load %mem[%i, %j - 1] : memref<10x10xf32>
       affine.store %0, %mem[%i, %j] : memref<10x10xf32>
    }
  }
  return
}

// CHECK-LABEL: func @negative_map_to_gpu_imperfect_nest_dep
func.func @negative_map_to_gpu_imperfect_nest_dep(%mem: memref<10x10xf32>) {
  /// Imperfect nest: The first inner loop 'j' has a dependency and is mapped
  /// to a thread dimension. This prevents parallelization of the parent loop.

  // CHECK-NOT: gpu.launch
  // CHECK: affine.for
  affine.for %i = 0 to 10 {
    // CHECK: affine.for
    affine.for %j = 1 to 10 {
       %0 = affine.load %mem[%i, %j - 1] : memref<10x10xf32>
       affine.store %0, %mem[%i, %j] : memref<10x10xf32>
    }
    // CHECK: affine.for
    affine.for %k = 0 to 10 {
       %1 = affine.load %mem[%i, %k] : memref<10x10xf32>
       affine.store %1, %mem[%i, %k] : memref<10x10xf32>
    }
  }
  return
}

// CHECK-LABEL: func @mixed_parallel_and_seq_siblings
func.func @mixed_parallel_and_seq_siblings(%mem: memref<10x10xf32>) {
  /// Sibling top-level loops are analyzed independently. The first nest is
  /// safe; the second has a dependency in a mapped loop (thread dim).

  // CHECK: gpu.launch
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
       %0 = affine.load %mem[%i, %j] : memref<10x10xf32>
       affine.store %0, %mem[%i, %j] : memref<10x10xf32>
    }
  }

  // CHECK-NOT: gpu.launch
  // CHECK: affine.for
  affine.for %i2 = 0 to 10 {
    // CHECK: affine.for
    affine.for %j2 = 1 to 10 {
       %1 = affine.load %mem[%i2, %j2 - 1] : memref<10x10xf32>
       affine.store %1, %mem[%i2, %j2] : memref<10x10xf32>
    }
  }
  return
}
