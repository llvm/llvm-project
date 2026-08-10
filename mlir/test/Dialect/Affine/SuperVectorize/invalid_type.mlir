// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=128" -split-input-file | FileCheck %s

// CHECK-LABEL: func @invalid_operand
func.func @invalid_operand(%a : vector<4xf32>, %b : vector<4xf32>) {
// CHECK: affine.for %{{.*}} = 0 to 10
// CHECK:   %{{.*}} = vector.reduction <add>, %{{.*}} : vector<4xf32> into f32
// CHECK: }
// CHECK: return
  affine.for %j = 0 to 10 {
    %1 = vector.reduction <add>, %a : vector<4xf32> into f32
  }
  return
}

// CHECK-LABEL: func @invalid_result
func.func @invalid_result(%a : memref<10x20xf32>, %b : memref<10x20xf32>) {
// CHECK: affine.for %{{.*}} = 0 to 10
// CHECK:   affine.for %{{.*}} = 0 to 5
// CHECK:     %{{.*}} = affine.vector_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x20xf32>, vector<4xf32>
// CHECK:     affine.vector_store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x20xf32>, vector<4xf32>
// CHECK:   }
// CHECK: }
// CHECK: return
  affine.for %j = 0 to 10 {
    affine.for %i = 0 to 5 {
      %ld0 = affine.vector_load %a[%j, %i] : memref<10x20xf32>, vector<4xf32>
      affine.vector_store %ld0, %b[%j, %i] : memref<10x20xf32>, vector<4xf32>
    }
  }
  return
}
