// RUN: mlir-opt %s -convert-linalg-tiled-loops-to-scf | FileCheck %s


#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func @tiled_loop(%A: memref<192x192xf32>,
                 %B: memref<192x192xf32>,
                 %C: memref<192x192xf32>) {
  %cst = constant 0.000000e+00 : f32
  %c24 = constant 24 : index
  %c16 = constant 16 : index
  %c0 = constant 0 : index
  %c192 = constant 192 : index

  linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B:  memref<192x192xf32>)
      outs (%C_ = %C: memref<192x192xf32>) {
    %0 = affine.min #map0(%i)
    %1 = memref.subview %A_[%i, 0] [%0, 192] [1, 1]
      : memref<192x192xf32> to memref<?x192xf32, #map1>
    %2 = affine.min #map2(%j)
    %3 = memref.subview %B_[0, %j] [192, %2] [1, 1]
      : memref<192x192xf32> to memref<192x?xf32, #map1>
    %4 = memref.subview %C_[%i, %j] [%0, %2] [1, 1]
      : memref<192x192xf32> to memref<?x?xf32, #map1>
    linalg.fill(%cst, %4) : f32, memref<?x?xf32, #map1>
    linalg.matmul ins(%1, %3 : memref<?x192xf32, #map1>,
                               memref<192x?xf32, #map1>)
                  outs(%4 : memref<?x?xf32, #map1>)
    linalg.yield
  }
  return
}

// CHECK-LABEL: @tiled_loop
// CHECK-SAME:  %[[A:.*]]: memref<192x192xf32>, %[[B:.*]]: memref<192x192xf32>,
// CHECK-SAME:  %[[C:.*]]: memref<192x192xf32>) {
// CHECK:       %[[C24:.*]] = constant 24 : index
// CHECK:       %[[C16:.*]] = constant 16 : index
// CHECK:       %[[C0:.*]] = constant 0 : index
// CHECK:       %[[C192:.*]] = constant 192 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C192]] step %[[C24]] {
// CHECK:         scf.for %[[J:.*]] = %[[C0]] to %[[C192]] step %[[C16]] {
// CHECK:           %[[A_sub:.*]] = memref.subview %[[A]][%[[I]]
// CHECK:           %[[B_sub:.*]] = memref.subview %[[B]][0, %[[J]]]
// CHECK:           %[[C_sub:.*]] = memref.subview %[[C]][%[[I]]
// CHECK:           linalg.fill
// CHECK:           linalg.matmul


func @tiled_loop_reduction(%A: memref<192x192xf32>,
                           %B: memref<192x192xf32>,
                           %C: memref<f32>) {
   %c24 = constant 24 : index
   %c16 = constant 16 : index
   %c0 = constant 0 : index
   %c192 = constant 192 : index
   %cst = constant 0.000000e+00 : f32

  linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B:  memref<192x192xf32>)
      outs (%C_ = %C: memref<f32>)
      iterators["reduction", "reduction"] {
    linalg.fill(%cst, %A_) : f32, memref<192x192xf32>
    linalg.yield
  }
  return
}

// CHECK-LABEL: @tiled_loop_reduction
// CHECK:       %[[C24:.*]] = constant 24 : index
// CHECK:       %[[C16:.*]] = constant 16 : index
// CHECK:       %[[C0:.*]] = constant 0 : index
// CHECK:       %[[C192:.*]] = constant 192 : index
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C24]]
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C16]]
// CHECK:           linalg.fill
