// RUN: mlir-opt %s -split-input-file | FileCheck %s

// -----

// Test with just loop IVs.
func @test0(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0, %i1] : memref<100x100xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test with loop IVs and constants.
func @test1(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
      affine.store %1, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}} + 3, %{{.*}} + 7] : memref<100x100xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} + 3, %{{.*}} + 7] : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test with loop IVs and function args without 'symbol' keyword (should
// be parsed as dim identifiers).
func @test2(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + %arg0, %i1 + %arg1] : memref<100x100xf32>
      affine.store %1, %0[%i0 + %arg0, %i1 + %arg1] : memref<100x100xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}} + %{{.*}}, %{{.*}} + %{{.*}}] : memref<100x100xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} + %{{.*}}, %{{.*}} + %{{.*}}] : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test with loop IVs and function args with 'symbol' keyword (should
// be parsed as symbol identifiers).
func @test3(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)]
        : memref<100x100xf32>
      affine.store %1, %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)]
        : memref<100x100xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}), %{{.*}} + symbol(%{{.*}})] : memref<100x100xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} + symbol(%{{.*}}), %{{.*}} + symbol(%{{.*}})] : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test with loop IVs, symbols and constants in nested affine expressions.
func @test4(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[(%i0 + symbol(%arg0)) floordiv 3 + 11,
                          (%i1 + symbol(%arg1)) mod 4 + 7] : memref<100x100xf32>
      affine.store %1, %0[(%i0 + symbol(%arg0)) floordiv 3 + 11,
                          (%i1 + symbol(%arg1)) mod 4 + 7] : memref<100x100xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[(%{{.*}} + symbol(%{{.*}})) floordiv 3 + 11, (%{{.*}} + symbol(%{{.*}})) mod 4 + 7] : memref<100x100xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[(%{{.*}} + symbol(%{{.*}})) floordiv 3 + 11, (%{{.*}} + symbol(%{{.*}})) mod 4 + 7] : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test with swizzled loop IVs.
func @test5(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<10x10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %1 = affine.load %0[%i2, %i0, %i1] : memref<10x10x10xf32>
        affine.store %1, %0[%i2, %i0, %i1] : memref<10x10x10xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// -----

// Test with swizzled loop IVs, duplicate args, and function args used as dims.
// Dim identifiers are assigned in parse order:
// d0 = %i2, d1 = %arg0, d2 = %i0, d3 = %i1, d4 = %arg1
func @test6(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<10x10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %1 = affine.load %0[%i2 + %arg0, %i0 + %i1, %i1 + %arg0 + %arg1]
          : memref<10x10x10xf32>
        affine.store %1, %0[%i2 + %arg0, %i0 + %i1, %i1 + %arg0 + %arg1]
          : memref<10x10x10xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}} + %{{.*}}, %{{.*}} + %{{.*}}, %{{.*}} + %{{.*}} + %{{.*}}] : memref<10x10x10xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} + %{{.*}}, %{{.*}} + %{{.*}}, %{{.*}} + %{{.*}} + %{{.*}}] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// -----

// Test with swizzled loop IVs, duplicate args, and function args used as syms.
// Dim and symbol identifiers are assigned in parse order:
// d0 = %i2, d1 = %i0, d2 = %i1
// s0 = %arg0, s1 = %arg1
func @test6(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<10x10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %1 = affine.load %0[%i2 + symbol(%arg0),
                            %i0 + %i1,
                            %i1 + symbol(%arg0) + symbol(%arg1)]
                              : memref<10x10x10xf32>
        affine.store %1, %0[%i2 + symbol(%arg0),
                             %i0 + %i1,
                             %i1 + symbol(%arg0) + symbol(%arg1)]
                              : memref<10x10x10xf32>
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}), %{{.*}} + %{{.*}}, %{{.*}} + symbol(%{{.*}}) + symbol(%{{.*}})] : memref<10x10x10xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} + symbol(%{{.*}}), %{{.*}} + %{{.*}}, %{{.*}} + symbol(%{{.*}}) + symbol(%{{.*}})] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0 + 1)>

// Test with operands without special SSA name.
func @test7() {
  %0 = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = affine.apply affine_map<(d1) -> (d1 + 1)>(%i0)
    %2 = affine.load %0[%1] : memref<10xf32>
    affine.store %2, %0[%1] : memref<10xf32>
// CHECK: affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  }
  return
}

// -----

// Test with zero-dimensional operands.
func @zero_dim(%arg0 : memref<i32>, %arg1 : memref<i32>) {
  %0 = affine.load %arg0[] : memref<i32>
  affine.store %0, %arg1[] : memref<i32>
  // CHECK: affine.load %{{.*}}[] : memref<i32>
  // CHECK: affine.store %{{.*}}, %{{.*}}[] : memref<i32>
  return
}

// -----

// Test with loop IVs and constants.
func @test_prefetch(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
      affine.prefetch %0[%i0 + 3, %i1 + 11], write, locality<0>, data : memref<100x100xf32>
      // CHECK: affine.prefetch %{{.*}}[%{{.*}} + 3, %{{.*}} + 11], write, locality<0>, data : memref<100x100xf32>
      affine.prefetch %0[%i0, %i1 + 1], read, locality<3>, instr : memref<100x100xf32>
      // CHECK: affine.prefetch %{{.*}}[%{{.*}}, %{{.*}} + 1], read, locality<3>, instr : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test with just loop IVs.
func @vector_load_vector_store_iv() {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 16 {
    affine.for %i1 = 0 to 16 step 8 {
      %1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<8xf32>
      affine.vector_store %1, %0[%i0, %i1] : memref<100x100xf32>, vector<8xf32>
// CHECK:      %[[buf:.*]] = alloc
// CHECK-NEXT: affine.for %[[i0:.*]] = 0
// CHECK-NEXT:   affine.for %[[i1:.*]] = 0
// CHECK-NEXT:     %[[val:.*]] = affine.vector_load %[[buf]][%[[i0]], %[[i1]]] : memref<100x100xf32>, vector<8xf32>
// CHECK-NEXT:     affine.vector_store %[[val]], %[[buf]][%[[i0]], %[[i1]]] : memref<100x100xf32>, vector<8xf32>
    }
  }
  return
}

// -----

// Test with loop IVs and constants.
func @vector_load_vector_store_iv_constant() {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 16 step 4 {
      %1 = affine.vector_load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<4xf32>
      affine.vector_store %1, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<4xf32>
// CHECK:      %[[buf:.*]] = alloc
// CHECK-NEXT: affine.for %[[i0:.*]] = 0
// CHECK-NEXT:   affine.for %[[i1:.*]] = 0
// CHECK-NEXT:     %[[val:.*]] = affine.vector_load %{{.*}}[%{{.*}} + 3, %{{.*}} + 7] : memref<100x100xf32>, vector<4xf32>
// CHECK-NEXT:     affine.vector_store %[[val]], %[[buf]][%[[i0]] + 3, %[[i1]] + 7] : memref<100x100xf32>, vector<4xf32>
    }
  }
  return
}

// -----

func @vector_load_vector_store_2d() {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 16 step 2{
    affine.for %i1 = 0 to 16 step 8 {
      %1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
      affine.vector_store %1, %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
// CHECK:      %[[buf:.*]] = alloc
// CHECK-NEXT: affine.for %[[i0:.*]] = 0
// CHECK-NEXT:   affine.for %[[i1:.*]] = 0
// CHECK-NEXT:     %[[val:.*]] = affine.vector_load %[[buf]][%[[i0]], %[[i1]]] : memref<100x100xf32>, vector<2x8xf32>
// CHECK-NEXT:     affine.vector_store %[[val]], %[[buf]][%[[i0]], %[[i1]]] : memref<100x100xf32>, vector<2x8xf32>
    }
  }
  return
}
