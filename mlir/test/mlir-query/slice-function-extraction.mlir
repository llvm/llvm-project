// RUN: mlir-query %s -c "m getDefinitionsByPredicate(hasOpName(\"memref.store\"),hasOpName(\"memref.alloc\"),true,false,false).extract(\"backward_slice\")" | FileCheck %s

// CHECK:       func.func @backward_slice(%{{.*}}: memref<10xf32>) -> (f32, index, index, f32, index, index, f32) {
// CHECK:         %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[I0:.*]] = affine.apply affine_map<()[s0] -> (s0)>()[%[[C0]]]
// CHECK-NEXT:    memref.store %[[CST0]], %{{.*}}[%[[I0]]] : memref<10xf32>
// CHECK-NEXT:    %[[CST2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[I1:.*]] = affine.apply affine_map<() -> (0)>()
// CHECK-NEXT:    memref.store %[[CST2]], %{{.*}}[%[[I1]]] : memref<10xf32>
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load %{{.*}}[%[[C1]]] : memref<10xf32>
// CHECK-NEXT:    memref.store %[[LOAD]], %{{.*}}[%[[C1]]] : memref<10xf32>
// CHECK-NEXT:    return %[[CST0]], %[[C0]], %[[I0]], %[[CST2]], %[[I1]], %[[C1]], %[[LOAD]] : f32, index, index, f32, index, index, f32

func.func @slicing_memref_store_trivial() {
  %0 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i1 = 0 to 10 {
    %1 = affine.apply affine_map<()[s0] -> (s0)>()[%c0]
    memref.store %cst, %0[%1] : memref<10xf32>
    %2 = memref.load %0[%c0] : memref<10xf32>
    %3 = affine.apply affine_map<()[] -> (0)>()[]
    memref.store %cst, %0[%3] : memref<10xf32>
    memref.store %2, %0[%c0] : memref<10xf32>
  }
  return
}
