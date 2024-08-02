// RUN: mlir-opt %s --sparse-reinterpret-map --sparsification="sparse-emit-strategy=debug-interface" --canonicalize --cse --allow-unregistered-dialect | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>



// CHECK-LABEL:   func.func @conv2d_all_sparse_CSR(
// CHECK:           "ne_sub<trivial<compressed[0,0]>>.begin"
// CHECK:           scf.while {{.*}} {
// CHECK:             "ne_sub<trivial<compressed[0,0]>>.not_end"
// CHECK:           } do {
// CHECK:             %[[D0:.*]] = "ne_sub<trivial<compressed[0,0]>>.deref"
// CHECK:             "ne_sub<trivial<compressed[0,1]>>.begin"
// CHECK:             scf.while {{.*}} {
// CHECK:               "ne_sub<trivial<compressed[0,1]>>.not_end"
// CHECK:             } do {
// CHECK:               %[[D1:.*]] = "ne_sub<trivial<compressed[0,1]>>.deref"
// CHECK:               "subsect<trivial<compressed[0,0]>>.begin"
// CHECK:               scf.while {{.*}} {
// CHECK:                 "subsect<trivial<compressed[0,0]>>.not_end
// CHECK:               } do {
// CHECK:                 %[[D2:.*]] = "subsect<trivial<compressed[0,0]>>.deref"
// CHECK:                 "trivial<batch[1,0]>.locate"(%{{.*}}, %[[D2]])
// CHECK:                 "subsect<trivial<compressed[0,1]>>.begin"
// CHECK:                 scf.while {{.*}} {
// CHECK:                   "subsect<trivial<compressed[0,1]>>.not_end"
// CHECK:                 } do {
// CHECK:                   %[[D3:.*]] = "subsect<trivial<compressed[0,1]>>.deref"
// CHECK:                   "trivial<batch[1,1]>.locate"(%{{.*}}, %[[D3]])
// CHECK:                   tensor.extract %{{.*}}{{\[}}%[[D2]], %[[D3]]]
// CHECK:                   arith.muli
// CHECK:                   arith.addi
// CHECK:                   "subsect<trivial<compressed[0,1]>>.next
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 "subsect<trivial<compressed[0,0]>>.next
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.if {{.*}} {
// CHECK:                 tensor.insert %{{.*}} into %{{.*}}{{\[}}%[[D0]], %[[D1]]]
// CHECK:                 scf.yield
// CHECK:               } else {
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               "ne_sub<trivial<compressed[0,1]>>.next"
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             "ne_sub<trivial<compressed[0,0]>>.next"
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           sparse_tensor.load
// CHECK:           return
// CHECK:         }
func.func @conv2d_all_sparse_CSR(%arg0: tensor<8x8xi32, #DCSR>,
                                 %arg1: tensor<3x3xi32>) -> tensor<6x6xi32, #DCSR> {
  %0 = tensor.empty() : tensor<6x6xi32, #DCSR>
  %1 = linalg.generic {
         indexing_maps = [#map, #map1, #map2],
         iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
         ins(%arg0, %arg1 : tensor<8x8xi32, #DCSR>, tensor<3x3xi32>)
         outs(%0 : tensor<6x6xi32, #DCSR>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %2 = arith.muli %in, %in_0 : i32
      %3 = arith.addi %out, %2 : i32
      linalg.yield %3 : i32
    } -> tensor<6x6xi32, #DCSR>
  return %1 : tensor<6x6xi32, #DCSR>
}
