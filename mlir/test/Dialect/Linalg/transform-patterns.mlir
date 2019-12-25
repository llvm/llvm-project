// RUN: mlir-opt %s -test-linalg-transform-patterns | FileCheck %s

// CHECK-DAG: #[[STRIDED_1D:.*]] = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[STRIDED_2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// CHECK-DAG: #[[mk:.*]] = (d0, d1, d2) -> (d0, d2)
// CHECK-DAG: #[[kn:.*]] = (d0, d1, d2) -> (d2, d1)
// CHECK-DAG: #[[mn:.*]] = (d0, d1, d2) -> (d0, d1)
// CHECK-DAG: #[[nm:.*]] = (d0, d1, d2) -> (d1, d0)
// CHECK-DAG: #[[km:.*]] = (d0, d1, d2) -> (d2, d0)

func @dot(%x: memref<?xf32, offset: ?, strides: [1]>,
          %y: memref<?xf32, offset: ?, strides: [1]>,
          %v: memref<f32>) {
  linalg.dot(%x, %y, %v) : memref<?xf32, offset: ?, strides: [1]>,
                           memref<?xf32, offset: ?, strides: [1]>,
                           memref<f32>
  return
}
// CHECK-LABEL: func @dot
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c8:.*]] = constant 8 : index
// CHECK-DAG  :   %[[c8000:.*]] = constant 8000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c8000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c8]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c1]] {
// CHECK      :         load
// CHECK      :         load
// CHECK      :         mulf
// CHECK      :         load
// CHECK      :         addf
// CHECK      :         store

func @matvec(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %x: memref<?xf32, offset: ?, strides: [1]>,
             %y: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec(%A, %x, %y) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?xf32, offset: ?, strides: [1]>,
                              memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @matvec
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c5:.*]] = constant 5 : index
// CHECK-DAG  :   %[[c6:.*]] = constant 6 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c5]]
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c6]]
// CHECK      :       linalg.matvec({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?xf32, #[[STRIDED_1D]]>, memref<?xf32, #[[STRIDED_1D]]>

func @matmul(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c2:.*]] = constant 2 : index
// CHECK-DAG  :   %[[c3:.*]] = constant 3 : index
// CHECK-DAG  :   %[[c4:.*]] = constant 4 : index
// CHECK-DAG  :   %[[c20:.*]] = constant 20 : index
// CHECK-DAG  :   %[[c30:.*]] = constant 30 : index
// CHECK-DAG  :   %[[c40:.*]] = constant 40 : index
// CHECK-DAG  :   %[[c200:.*]] = constant 200 : index
// CHECK-DAG  :   %[[c300:.*]] = constant 300 : index
// CHECK-DAG  :   %[[c400:.*]] = constant 400 : index
// CHECK-DAG  :   %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG  :   %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG  :   %[[c4000:.*]] = constant 4000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK      :         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK      :           loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK      :             loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK      :               loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK      :                 loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK      :                   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK      :                     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2]] {
// CHECK      :                       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3]] {
// CHECK      :                         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4]] {
// CHECK      :                           linalg.matmul({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

#some_generic_trait = {
  args_in = 1,
  args_out = 1,
  indexing_maps = [
    (i, j) -> (i, j),
    (i, j) -> (i, j)
  ],
  iterator_types = ["parallel", "parallel"]
}
func @fusion_test(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %C: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %D: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %E: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  // This should not be fused as it would violate dependencies. It will get
  // tiled for all levels of the memory hierarchy.
  linalg.matmul(%A, %A, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  // This should be fused.
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  // This should not be fused or transformed at all since there are no patterns
  // on it. However it will be reordered because there are no dependencies.
  linalg.generic #some_generic_trait %A, %D {
    ^bb(%a: f32, %b: f32) :
      linalg.yield %a : f32
  } : memref<?x?xf32, offset: ?, strides: [?, 1]>,
      memref<?x?xf32, offset: ?, strides: [?, 1]>

  linalg.matmul(%C, %D, %E) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  return
}
// CHECK-LABEL: func @fusion_test
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c2:.*]] = constant 2 : index
// CHECK-DAG  :   %[[c3:.*]] = constant 3 : index
// CHECK-DAG  :   %[[c4:.*]] = constant 4 : index
// CHECK-DAG  :   %[[c20:.*]] = constant 20 : index
// CHECK-DAG  :   %[[c30:.*]] = constant 30 : index
// CHECK-DAG  :   %[[c40:.*]] = constant 40 : index
// CHECK-DAG  :   %[[c100:.*]] = constant 100 : index
// CHECK-DAG  :   %[[c150:.*]] = constant 150 : index
// CHECK-DAG  :   %[[c200:.*]] = constant 200 : index
// CHECK-DAG  :   %[[c300:.*]] = constant 300 : index
// CHECK-DAG  :   %[[c400:.*]] = constant 400 : index
// CHECK-DAG  :   %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG  :   %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG  :   %[[c4000:.*]] = constant 4000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK      :         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK      :           loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK      :             loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK      :               loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK      :                 loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK      :                   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK      :                     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2]] {
// CHECK      :                       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3]] {
// CHECK      :                         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4]] {
// CHECK      :                           linalg.matmul({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>
//
// CHECK      :   linalg.generic
//
// CHECK      :   loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c100]] {
// CHECK      :     loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c150]] {
// CHECK      :       loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c2]] {
// CHECK      :         loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c3]] {
// CHECK      :           loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c4]] {
// CHECK      :             linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>
// CHECK      :       loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c2]] {
// CHECK      :         loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c3]] {
// CHECK      :           loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c4]] {
// CHECK      :             linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    (m, n, k) -> (m, k),
    (m, n, k) -> (k, n),
    (m, n, k) -> (m, n)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  __internal_linalg_transform__ = "_marked_matmul_"
}
func @vectorization_test(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  linalg.generic #matmul_trait %A, %B, %C {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = mulf %a, %b: f32
      %e = addf %c, %d: f32
      linalg.yield %e : f32
  } : memref<8x16xf32>, memref<16x32xf32>, memref<8x32xf32>
  return
}

// CHECK-LABEL: func @vectorization_test
//       CHECK: vector.type_cast %{{.*}} : memref<8x16xf32> to memref<vector<8x16xf32>>
//       CHECK: load %{{.*}}[] : memref<vector<8x16xf32>>
//       CHECK: vector.type_cast %{{.*}} : memref<16x32xf32> to memref<vector<16x32xf32>>
//       CHECK: load %{{.*}}[] : memref<vector<16x32xf32>>
//       CHECK: vector.type_cast %{{.*}} : memref<8x32xf32> to memref<vector<8x32xf32>>
//       CHECK: load %{{.*}}[] : memref<vector<8x32xf32>>
//       CHECK: vector.contract {indexing_maps = [#[[mk]], #[[kn]], #[[mn]]], iterator_types = ["parallel", "parallel", "reduction"]} %{{.*}}, %{{.*}}, %{{.*}} : vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>
//       CHECK: store %{{.*}}, %{{.*}}[] : memref<vector<8x32xf32>>
func @fma(%a: f32, %b: f32, %c: f32) -> f32 {
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          return %e: f32
        }
#matmul_accesses = [
          (m, n, k) -> (m, k),
          (m, n, k) -> (k, n),
          (m, n, k) -> (m, n)
]
#generic_matmul_trait = {
          args_in = 2,
          args_out = 1,
          fun = @fma,
          indexing_maps = #matmul_accesses,
          library_call = "linalg_matmul",
          iterator_types = ["parallel", "parallel", "reduction"]
        }

func @permute_generic(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
           %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
           %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.generic #generic_matmul_trait %A, %B, %C : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>

  return
}
// CHECK-LABEL : func @fma
// CHECK-LABEL : func @permute_generic
// CHECK       : linalg.generic {args_in = 2, args_out = 1, fun = @fma, indexing_maps = [#[[kn]], #[[nm]], #[[km]]], iterator_types = ["parallel", "reduction", "parallel"], library_call = "linalg_matmul"} %{{.*}}, %{{.*}}, %{{.*}} : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

func @fma_indexed(%i: index, %j: index, %k: index, %a: f32, %b: f32, %c: f32) -> f32 {
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          return %e: f32
}
#indexed_matmul_trait = {
          args_in = 2,
          args_out = 1,
          fun = @fma_indexed,
          indexing_maps = #matmul_accesses,
          library_call = "linalg_matmul_indexed",
          iterator_types = ["parallel", "parallel", "reduction"]
}
func @permute_generic_indexed(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
           %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
           %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.indexed_generic #indexed_matmul_trait %A, %B, %C : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL : func @fma_indexed
// CHECK-LABEL : func @permute_generic_indexed
// CHECK       : linalg.indexed_generic {args_in = 2, args_out = 1, fun = @fma, indexing_maps = [#[[kn]], #[[nm]], #[[km]]], iterator_types = ["parallel", "reduction", "parallel"], library_call = "linalg_matmul_indexed"} %{{.*}}, %{{.*}}, %{{.*}} : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

func @dot_perm(%x: memref<?xf32, offset: ?, strides: [1]>,
          %y: memref<?xf32, offset: ?, strides: [1]>,
          %v: memref<f32>) {
  linalg.dot(%x, %y, %v) {__internal_linalg_transform__ = "__with_perm__"} :
            memref<?xf32, offset: ?, strides: [1]>,
            memref<?xf32, offset: ?, strides: [1]>,
            memref<f32>
  return
}
// CHECK-LABEL: func @dot_perm
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c8:.*]] = constant 8 : index
// CHECK-DAG  :   %[[c8000:.*]] = constant 8000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c8000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c8]] {
// CHECK      :       linalg.dot({{.*}}, {{.*}}, {{.*}}) : memref<?xf32, #[[STRIDED_1D]]>, memref<?xf32, #[[STRIDED_1D]]>, memref<f32>

func @matvec_perm(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %x: memref<?xf32, offset: ?, strides: [1]>,
             %y: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec(%A, %x, %y) {__internal_linalg_transform__ = "__with_perm__"} :
               memref<?x?xf32, offset: ?, strides: [?, 1]>,
               memref<?xf32, offset: ?, strides: [1]>,
               memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @matvec_perm
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c5:.*]] = constant 5 : index
// CHECK-DAG  :   %[[c6:.*]] = constant 6 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c6]]
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c5]]
// CHECK      :       linalg.matvec({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?xf32, #[[STRIDED_1D]]>, memref<?xf32, #[[STRIDED_1D]]>

func @matmul_perm(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul(%A, %B, %C) {__internal_linalg_transform__ = "__with_perm__"} :
               memref<?x?xf32, offset: ?, strides: [?, 1]>,
               memref<?x?xf32, offset: ?, strides: [?, 1]>,
               memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul_perm
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c2:.*]] = constant 2 : index
// CHECK-DAG  :   %[[c3:.*]] = constant 3 : index
// CHECK-DAG  :   %[[c4:.*]] = constant 4 : index
// CHECK-DAG  :   %[[c20:.*]] = constant 20 : index
// CHECK-DAG  :   %[[c30:.*]] = constant 30 : index
// CHECK-DAG  :   %[[c40:.*]] = constant 40 : index
// CHECK-DAG  :   %[[c200:.*]] = constant 200 : index
// CHECK-DAG  :   %[[c300:.*]] = constant 300 : index
// CHECK-DAG  :   %[[c400:.*]] = constant 400 : index
// CHECK-DAG  :   %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG  :   %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG  :   %[[c4000:.*]] = constant 4000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK      :         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK      :           loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK      :             loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK      :               loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK      :                 loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK      :                   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK      :                           linalg.matmul({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

func @promote_subview_matmul(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                             %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                             %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  %c2000 = constant 2000 : index
  %c3000 = constant 3000 : index
  %c4000 = constant 4000 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = dim %arg0, 0 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %1 = dim %arg0, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %2 = dim %arg1, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  loop.for %arg3 = %c0 to %0 step %c2000 {
    loop.for %arg4 = %c0 to %2 step %c3000 {
      loop.for %arg5 = %c0 to %1 step %c4000 {
        %3 = std.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %4 = std.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %5 = std.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%3, %4, %5) {__internal_linalg_transform__ = "_promote_views_"} :
                      memref<?x?xf32, offset: ?, strides: [?, ?]>,
                      memref<?x?xf32, offset: ?, strides: [?, ?]>,
                      memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return
}
// CHECK-LABEL: func @promote_subview_matmul
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK      :         %[[s0:.*]] = std.subview {{%.*}}[{{%.*}}, {{%.*}}][{{%.*}}, {{%.*}}][{{%.*}}, {{%.*}}] : memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK      :         %[[s1:.*]] = std.subview {{%.*}}[{{%.*}}, {{%.*}}][{{%.*}}, {{%.*}}][{{%.*}}, {{%.*}}] : memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK      :         %[[s2:.*]] = std.subview {{%.*}}[{{%.*}}, {{%.*}}][{{%.*}}, {{%.*}}][{{%.*}}, {{%.*}}] : memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK      :         %[[a0:.*]] = alloc({{%.*}}) : memref<?xi8>
// CHECK      :         %[[v0:.*]] = std.view %[[a0]][][{{%.*}}, {{%.*}}]: memref<?xi8> to memref<?x?xf32>
// CHECK      :         %[[l0:.*]] = linalg.slice %[[v0]][{{%.*}}, {{%.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32, #map{{.*}}>
// CHECK      :         %[[a1:.*]] = alloc({{%.*}}) : memref<?xi8>
// CHECK      :         %[[v1:.*]] = std.view %[[a1]][][{{%.*}}, {{%.*}}]: memref<?xi8> to memref<?x?xf32>
// CHECK      :         %[[l1:.*]] = linalg.slice %[[v1]][{{%.*}}, {{%.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32, #map{{.*}}>
// CHECK      :         %[[a2:.*]] = alloc({{%.*}}) : memref<?xi8>
// CHECK      :         %[[v2:.*]] = std.view %[[a2]][][{{%.*}}, {{%.*}}]: memref<?xi8> to memref<?x?xf32>
// CHECK      :         %[[l2:.*]] = linalg.slice %[[v2]][{{%.*}}, {{%.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32, #map{{.*}}>
// CHECK      :         linalg.copy(%[[s0]], %[[l0]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK      :         linalg.copy(%[[s1]], %[[l1]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK      :         linalg.copy(%[[s2]], %[[l2]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK      :         linalg.matmul(%[[v0]], %[[v1]], %[[v2]]) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>
