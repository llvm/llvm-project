// RUN: mlir-opt %s -test-linalg-transform-patterns=test-patterns | FileCheck %s

// CHECK-DAG: #[[$STRIDED_1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// Map corresponding to a 2D memory access where the stride along the last dim is known to be 1.
// CHECK-DAG: #[[$STRIDED_2D_u_1:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// Map corresponding to a 2D memory access where the stride along all dims are unknown.
// CHECK-DAG: #[[$STRIDED_2D:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
// CHECK-DAG: #[[$kn:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$nm:.*]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[$km:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>

func @dot(%x: memref<?xf32, offset: ?, strides: [1]>,
          %y: memref<?xf32, offset: ?, strides: [1]>,
          %v: memref<f32>) {
  linalg.dot { __internal_linalg_transform__ = "MEM" }
    ins(%x, %y: memref<?xf32, offset: ?, strides: [1]>,
                memref<?xf32, offset: ?, strides: [1]>)
    outs(%v: memref<f32>)

  return
}
// CHECK-LABEL: func @dot
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c1:.*]] = constant 1 : index
// CHECK-DAG:     %[[c8000:.*]] = constant 8000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c8000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c1]] {
// CHECK:               load
// CHECK:               load
// CHECK:               load
// CHECK:               mulf
// CHECK:               addf
// CHECK:               store

func @matvec(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %x: memref<?xf32, offset: ?, strides: [1]>,
             %y: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec
    ins(%A, %x: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                memref<?xf32, offset: ?, strides: [1]>)
    outs(%y: memref<?xf32, offset: ?, strides: [1]>)
  return
}
// CHECK-LABEL: func @matvec
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c5:.*]] = constant 5 : index
// CHECK-DAG:     %[[c6:.*]] = constant 6 : index
// CHECK:         scf.parallel {{.*}} step (%[[c5]])
// CHECK:           scf.for {{.*}} step %[[c6]]
// CHECK:             linalg.matvec
// CHECK:               ins({{.*}}: memref<?x?xf32, #[[$STRIDED_2D_u_1]]>, memref<?xf32, #[[$STRIDED_1D]]>)
// CHECK:              outs({{.*}}: memref<?xf32, #[[$STRIDED_1D]]>)

func @matmul(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul { __internal_linalg_transform__ = "MEM" }
    ins(%A, %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                memref<?x?xf32, offset: ?, strides: [?, 1]>)
    outs(%C: memref<?x?xf32, offset: ?, strides: [?, 1]>)
  return
}
// CHECK-LABEL: func @matmul
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c2:.*]] = constant 2 : index
// CHECK-DAG:     %[[c3:.*]] = constant 3 : index
// CHECK-DAG:     %[[c4:.*]] = constant 4 : index
// CHECK-DAG:     %[[c20:.*]] = constant 20 : index
// CHECK-DAG:     %[[c30:.*]] = constant 30 : index
// CHECK-DAG:     %[[c40:.*]] = constant 40 : index
// CHECK-DAG:     %[[c200:.*]] = constant 200 : index
// CHECK-DAG:     %[[c300:.*]] = constant 300 : index
// CHECK-DAG:     %[[c400:.*]] = constant 400 : index
// CHECK-DAG:     %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = constant 4000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:               scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK:                 scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK:                   scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK:                     scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK:                       scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK:                         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK:                           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2]] {
// CHECK:                             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3]] {
// CHECK:                               scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4]] {
// CHECK:                                 linalg.matmul
// CHECK:                                   ins({{.*}}: memref<?x?xf32, #[[$STRIDED_2D_u_1]]>, memref<?x?xf32, #[[$STRIDED_2D_u_1]]>)
// CHECK:                                  outs({{.*}}: memref<?x?xf32, #[[$STRIDED_2D_u_1]]>)

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#generic_matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = #matmul_accesses,
  library_call = "linalg_matmul",
  iterator_types = ["parallel", "parallel", "reduction"]
}
func @permute_generic(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
           %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
           %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.generic #generic_matmul_trait
    ins(%A, %B : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                 memref<?x?xf32, offset: ?, strides: [?, 1]>)
   outs(%C : memref<?x?xf32, offset: ?, strides: [?, 1]>) {
    ^bb(%a: f32, %b: f32, %c: f32):
      %d = mulf %a, %b: f32
      %e = addf %c, %d: f32
      linalg.yield %e: f32
  }
  return
}
// CHECK-LABEL:  func @permute_generic
// CHECK:        linalg.generic {
// CHECK-SAME:   indexing_maps = [#[[$kn]], #[[$nm]], #[[$km]]],
// CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel"],
// CHECK-SAME:   library_call = "linalg_matmul"}
// CHECK:          memref<?x?xf32, #[[$STRIDED_2D_u_1]]>,
// CHECK-SAME:     memref<?x?xf32, #[[$STRIDED_2D_u_1]]>
// CHECK-SAME:     memref<?x?xf32, #[[$STRIDED_2D_u_1]]>

func @matvec_perm(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %x: memref<?xf32, offset: ?, strides: [1]>,
             %y: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec {__internal_linalg_transform__ = "__with_perm__"}
    ins(%A, %x: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                memref<?xf32, offset: ?, strides: [1]>)
   outs(%y: memref<?xf32, offset: ?, strides: [1]>)
  return
}
// CHECK-LABEL: func @matvec_perm
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c5:.*]] = constant 5 : index
// CHECK-DAG:     %[[c6:.*]] = constant 6 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c6]]
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c5]]
// CHECK:             linalg.matvec
// CHECK:               ins({{.*}}: memref<?x?xf32, #[[$STRIDED_2D_u_1]]>, memref<?xf32, #[[$STRIDED_1D]]>)
// CHECK:              outs({{.*}}: memref<?xf32, #[[$STRIDED_1D]]>)

func @matmul_perm(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul {__internal_linalg_transform__ = "__with_perm__"}
    ins(%A, %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                memref<?x?xf32, offset: ?, strides: [?, 1]>)
   outs(%C : memref<?x?xf32, offset: ?, strides: [?, 1]>)
  return
}
// CHECK-LABEL: func @matmul_perm
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c20:.*]] = constant 20 : index
// CHECK-DAG:     %[[c30:.*]] = constant 30 : index
// CHECK-DAG:     %[[c40:.*]] = constant 40 : index
// CHECK-DAG:     %[[c200:.*]] = constant 200 : index
// CHECK-DAG:     %[[c300:.*]] = constant 300 : index
// CHECK-DAG:     %[[c400:.*]] = constant 400 : index
// CHECK-DAG:     %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = constant 4000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:               scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK:                 scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK:                   scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK:                     scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK:                       scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK:                         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK:                                 linalg.matmul
// CHECK:                                  ins({{.*}}: memref<?x?xf32, #[[$STRIDED_2D_u_1]]>, memref<?x?xf32, #[[$STRIDED_2D_u_1]]>)
// CHECK:                                   outs({{.*}}: memref<?x?xf32, #[[$STRIDED_2D_u_1]]>)

func @promote_subview_matmul(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                             %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                             %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  %c2000 = constant 2000 : index
  %c3000 = constant 3000 : index
  %c4000 = constant 4000 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %2 = memref.dim %arg1, %c1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  scf.for %arg3 = %c0 to %0 step %c2000 {
    scf.for %arg4 = %c0 to %2 step %c3000 {
      scf.for %arg5 = %c0 to %1 step %c4000 {
        %3 = memref.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %4 = memref.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %5 = memref.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul {__internal_linalg_transform__ = "_promote_views_"}
          ins(%3, %4: memref<?x?xf32, offset: ?, strides: [?, ?]>,
                      memref<?x?xf32, offset: ?, strides: [?, ?]>)
         outs(%5: memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return
}
// CHECK-LABEL: func @promote_subview_matmul
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = constant 4000 : index
// CHECK:         scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:           scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:             scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:               %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:               %[[s1:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:               %[[s2:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:               %[[a0:.*]] = memref.alloc() : memref<32000000xi8>
// CHECK:               %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<32000000xi8> to memref<?x?xf32>
// CHECK:               %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1]
// CHECK-SAME:            memref<?x?xf32> to memref<?x?xf32, #[[$STRIDED_2D_u_1]]>
// CHECK:               %[[a1:.*]] = memref.alloc() : memref<48000000xi8>
// CHECK:               %[[v1:.*]] = memref.view %[[a1]]{{.*}} : memref<48000000xi8> to memref<?x?xf32>
// CHECK:               %[[l1:.*]] = memref.subview %[[v1]][0, 0] [%{{.*}}, %{{.*}}] [1, 1]
// CHECK-SAME:            memref<?x?xf32> to memref<?x?xf32, #[[$STRIDED_2D_u_1]]>
// CHECK:               %[[a2:.*]] = memref.alloc() : memref<24000000xi8>
// CHECK:               %[[v2:.*]] = memref.view %[[a2]]{{.*}} : memref<24000000xi8> to memref<?x?xf32>
// CHECK:               %[[l2:.*]] = memref.subview %[[v2]][0, 0] [%{{.*}}, %{{.*}}] [1, 1]
// CHECK-SAME:            memref<?x?xf32> to memref<?x?xf32, #[[$STRIDED_2D_u_1]]>
// CHECK:               linalg.copy(%[[s0]], %[[l0]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK:               linalg.copy(%[[s1]], %[[l1]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK:               linalg.copy(%[[s2]], %[[l2]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK:               linalg.matmul
// CHECK-SAME:                 ins(%[[v0]], %[[v1]] : memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME:                outs(%[[v2]] : memref<?x?xf32>)

func @promote_first_subview_matmul(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                             %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                             %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  %c2000 = constant 2000 : index
  %c3000 = constant 3000 : index
  %c4000 = constant 4000 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %2 = memref.dim %arg1, %c1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  scf.for %arg3 = %c0 to %0 step %c2000 {
    scf.for %arg4 = %c0 to %2 step %c3000 {
      scf.for %arg5 = %c0 to %1 step %c4000 {
        %3 = memref.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %4 = memref.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %5 = memref.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul {__internal_linalg_transform__ = "_promote_first_view_"}
          ins(%3, %4: memref<?x?xf32, offset: ?, strides: [?, ?]>,
                      memref<?x?xf32, offset: ?, strides: [?, ?]>)
         outs(%5: memref<?x?xf32, offset: ?, strides: [?, ?]>)
      }
    }
  }
  return
}
// CHECK-LABEL: func @promote_first_subview_matmul
// CHECK-DAG:     %[[c0:.*]] = constant 0 : index
// CHECK-DAG:     %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG:     %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG:     %[[c4000:.*]] = constant 4000 : index
// CHECK:   scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK:     scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK:       scf.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK:         %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:         %[[s1:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:         %[[s2:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:         %[[a0:.*]] = memref.alloc() : memref<32000000xi8>
// CHECK:         %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<32000000xi8> to memref<?x?xf32>
// CHECK:         %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #[[$STRIDED_2D_u_1]]>
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.view
// CHECK-NOT:     memref.subview
// CHECK:         linalg.copy(%[[s0]], %[[l0]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK-NOT:     linalg.copy
// CHECK:         linalg.matmul
// CHECK-SAME:           ins(%[[v0]], %[[s1]] : memref<?x?xf32>, memref<?x?xf32, #[[$STRIDED_2D]]>)
// CHECK-SAME:          outs(%[[s2]] : memref<?x?xf32, #[[$STRIDED_2D]]>)

func @aligned_promote_fill(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  %c2000 = constant 2000 : index
  %c4000 = constant 4000 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cf = constant 1.0 : f32
  %3 = memref.subview %arg0[%c0, %c0][%c2000, %c4000][%c1, %c1] :
 	 memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
  linalg.fill(%cf, %3) { __internal_linalg_transform__ = "_promote_views_aligned_"}
  	: f32, memref<?x?xf32, offset: ?, strides: [?, ?]>
  return
}
// CHECK-LABEL: func @aligned_promote_fill
// CHECK:	  %[[cf:.*]] = constant {{.*}} : f32
// CHECK:         %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xf32, #map{{.*}}> to memref<?x?xf32, #map{{.*}}>
// CHECK:         %[[a0:.*]] = memref.alloc() {alignment = 32 : i64} : memref<32000000xi8>
// CHECK:         %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<32000000xi8> to memref<?x?xf32>
// CHECK:         %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #[[$STRIDED_2D_u_1]]>
// CHECK:         linalg.fill({{.*}}, %[[v0]]) : f32, memref<?x?xf32>
// CHECK:         linalg.copy(%[[s0]], %[[l0]]) : memref<?x?xf32, #map{{.*}}>, memref<?x?xf32, #map{{.*}}>
// CHECK:         linalg.fill(%[[cf]], %[[v0]]) : f32, memref<?x?xf32>

func @aligned_promote_fill_complex(%arg0: memref<?x?xcomplex<f32>, offset: ?, strides: [?, 1]>) {
  %c2000 = constant 2000 : index
  %c4000 = constant 4000 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cf = constant 1.0 : f32
  %cc = complex.create %cf, %cf : complex<f32>
  %3 = memref.subview %arg0[%c0, %c0][%c2000, %c4000][%c1, %c1] :
 	 memref<?x?xcomplex<f32>, offset: ?, strides: [?, 1]> to memref<?x?xcomplex<f32>, offset: ?, strides: [?, ?]>
  linalg.fill(%cc, %3) { __internal_linalg_transform__ = "_promote_views_aligned_"}
  	: complex<f32>, memref<?x?xcomplex<f32>, offset: ?, strides: [?, ?]>
  return
}
// CHECK-LABEL: func @aligned_promote_fill_complex
// CHECK:	  %[[cc:.*]] = complex.create {{.*}} : complex<f32>
// CHECK:         %[[s0:.*]] = memref.subview {{.*}}: memref<?x?xcomplex<f32>, #map{{.*}}> to memref<?x?xcomplex<f32>, #map{{.*}}>
// CHECK:         %[[a0:.*]] = memref.alloc() {alignment = 32 : i64} : memref<64000000xi8>
// CHECK:         %[[v0:.*]] = memref.view %[[a0]]{{.*}} : memref<64000000xi8> to memref<?x?xcomplex<f32>>
// CHECK:         %[[l0:.*]] = memref.subview %[[v0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xcomplex<f32>> to memref<?x?xcomplex<f32>, #[[$STRIDED_2D_u_1]]>
// CHECK:         linalg.fill({{.*}}, %[[v0]]) : complex<f32>, memref<?x?xcomplex<f32>>
// CHECK:         linalg.copy(%[[s0]], %[[l0]]) : memref<?x?xcomplex<f32>, #map{{.*}}>, memref<?x?xcomplex<f32>, #map{{.*}}>
// CHECK:         linalg.fill(%[[cc]], %[[v0]]) : complex<f32>, memref<?x?xcomplex<f32>>

func @tile_permute_parallel_loop(%arg0: memref<?x?xf32>,
                                 %arg1: memref<?x?xf32>,
                                 %arg2: memref<?x?xf32>) {
  linalg.matmul {__internal_linalg_transform__ = "par__with_perm__"}
    ins(%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
   outs(%arg2: memref<?x?xf32>)
  return
}
// CHECK-LABEL: func @tile_permute_parallel_loop
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C16:.*]] = constant 16 : index
//   CHECK-DAG:   %[[C8:.*]] = constant 8 : index
//   CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[D0:.*]] = memref.dim %[[ARG0]], %c0
//   CHECK-DAG:   %[[D1:.*]] = memref.dim %[[ARG0]], %c1
//   CHECK-DAG:   %[[D2:.*]] = memref.dim %[[ARG1]], %c1
//       CHECK:   scf.parallel (%{{.*}}) = (%[[C0]]) to (%[[D2]]) step (%[[C8]])
//       CHECK:     scf.for %{{.*}} = %[[C0]] to %[[D1]] step %[[C4]]
//       CHECK:       scf.parallel (%{{.*}}) = (%[[C0]]) to (%[[D0]]) step (%[[C16]])
