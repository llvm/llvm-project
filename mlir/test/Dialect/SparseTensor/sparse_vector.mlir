// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-SCALAR
// RUN: mlir-opt %s --sparse-reinterpret-map --sparse-reinterpret-map -sparsification -cse -sparse-vectorization="vl=16" -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC16
// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -cse -sparse-vectorization="vl=16 enable-simd-index32=true" -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC16-IDX32
// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -cse -sparse-vectorization="vl=4 enable-vla-vectorization=true" -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC4-SVE

#DenseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>

#trait_scale_d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b"
}

//
// CHECK-SCALAR-LABEL: func @scale_d
// CHECK-SCALAR-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-SCALAR-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-SCALAR-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-SCALAR:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] {
// CHECK-SCALAR:         %[[l:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-SCALAR:         %[[m:.*]] = arith.mulf %[[l]], %{{.*}} : f32
// CHECK-SCALAR:         store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>
// CHECK-SCALAR:       }
// CHECK-SCALAR:       return
//
// CHECK-VEC16-LABEL: func @scale_d
// CHECK-VEC16-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-VEC16:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] {
// CHECK-VEC16:         %[[r:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC16:         %[[b:.*]] = vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK-VEC16:         %[[m:.*]] = arith.mulf %[[r]], %[[b]] : vector<16xf32>
// CHECK-VEC16:         vector.store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC16:       }
// CHECK-VEC16:       return
//
// CHECK-VEC16-IDX32-LABEL: func @scale_d
// CHECK-VEC16-IDX32-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-IDX32-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-IDX32-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-VEC16-IDX32:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] {
// CHECK-VEC16-IDX32:         %[[r:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC16-IDX32:         %[[b:.*]] = vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK-VEC16-IDX32:         %[[m:.*]] = arith.mulf %[[r]], %[[b]] : vector<16xf32>
// CHECK-VEC16-IDX32:         vector.store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC16-IDX32:       }
// CHECK-VEC16-IDX32:       return
//
// CHECK-VEC4-SVE:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)
// CHECK-VEC4-SVE-LABEL: func @scale_d
// CHECK-VEC4-SVE-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC4-SVE-DAG:   %[[c4:.*]] = arith.constant 4 : index
// CHECK-VEC4-SVE-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-VEC4-SVE-DAG:   %[[v0:.*]] = arith.constant dense<0.000000e+00> : vector<[4]xf32>
// CHECK-VEC4-SVE-DAG:   %[[vscale:.*]] = vector.vscale
// CHECK-VEC4-SVE:       %[[step:.*]] = arith.muli %[[vscale]], %[[c4]] : index
// CHECK-VEC4-SVE:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[step]] {
// CHECK-VEC4-SVE:         %[[sub:.*]] = affine.min #[[$map]](%[[c1024]], %[[i]])[%[[step]]]
// CHECK-VEC4-SVE:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<[4]xi1>
// CHECK-VEC4-SVE:         %[[val:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %[[v0]] : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[scalev:.*]] = vector.broadcast %{{.*}} : f32 to vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[scaled:.*]] = arith.mulf %[[val]], %[[scalev]] : vector<[4]xf32>
// CHECK-VEC4-SVE:         vector.maskedstore %{{.*}}[%[[i]]], %[[mask]], %[[scaled]] : memref<1024xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK-VEC4-SVE:       }
// CHECK-VEC4-SVE:       return
//
func.func @scale_d(%arga: tensor<1024xf32, #DenseVector>, %b: f32, %argx: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = linalg.generic #trait_scale_d
    ins(%arga: tensor<1024xf32, #DenseVector>)
    outs(%argx: tensor<1024xf32>) {
      ^bb(%a: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#trait_mul_s = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b(i)"
}

//
// CHECK-SCALAR-LABEL: func @mul_s
// CHECK-SCALAR-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-SCALAR-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-SCALAR:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-SCALAR:       %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-SCALAR:       %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-SCALAR:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-SCALAR:       %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-SCALAR:       %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-SCALAR:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK-SCALAR:         %[[li:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-SCALAR:         %[[zi:.*]] = arith.extui %[[li]] : i32 to i64
// CHECK-SCALAR:         %[[ci:.*]] = arith.index_cast %[[zi]] : i64 to index
// CHECK-SCALAR:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-SCALAR:         %[[lb:.*]] = memref.load %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK-SCALAR:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : f32
// CHECK-SCALAR:         store %[[m]], %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK-SCALAR:       }
// CHECK-SCALAR:       return
//
// CHECK-VEC16:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC16-LABEL: func @mul_s
// CHECK-VEC16-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC16-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC16:       %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-VEC16:       %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-VEC16:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC16:       %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-VEC16:       %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-VEC16:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC16:         %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[i]])[%[[c16]]]
// CHECK-VEC16:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC16:         %[[li:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC16:         %[[zi:.*]] = arith.extui %[[li]] : vector<16xi32> to vector<16xi64>
// CHECK-VEC16:         %[[la:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16:         %[[lb:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[zi]]], %[[mask]], %{{.*}} : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC16:         vector.scatter %{{.*}}[%[[c0]]] [%[[zi]]], %[[mask]], %[[m]] : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32>
// CHECK-VEC16:       }
// CHECK-VEC16:       return
//
// CHECK-VEC16-IDX32:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC16-IDX32-LABEL: func @mul_s
// CHECK-VEC16-IDX32-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-IDX32-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC16-IDX32-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-IDX32:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC16-IDX32:       %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-VEC16-IDX32:       %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-VEC16-IDX32:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC16-IDX32:       %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-VEC16-IDX32:       %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-VEC16-IDX32:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC16-IDX32:         %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[i]])[%[[c16]]]
// CHECK-VEC16-IDX32:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC16-IDX32:         %[[li:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC16-IDX32:         %[[la:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16-IDX32:         %[[lb:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[li]]], %[[mask]], %{{.*}} : memref<1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16-IDX32:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC16-IDX32:         vector.scatter %{{.*}}[%[[c0]]] [%[[li]]], %[[mask]], %[[m]] : memref<1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
// CHECK-VEC16-IDX32:       }
// CHECK-VEC16-IDX32:       return
//
// CHECK-VEC4-SVE:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)
// CHECK-VEC4-SVE-LABEL: func @mul_s
// CHECK-VEC4-SVE-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC4-SVE-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC4-SVE-DAG:   %[[c4:.*]] = arith.constant 4 : index
// CHECK-VEC4-SVE-DAG:   %[[v0i:.*]] = arith.constant dense<0> : vector<[4]xi32>
// CHECK-VEC4-SVE-DAG:   %[[v0f:.*]] = arith.constant dense<0.000000e+00> : vector<[4]xf32>
// CHECK-VEC4-SVE:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC4-SVE:       %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-VEC4-SVE:       %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-VEC4-SVE:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC4-SVE:       %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-VEC4-SVE:       %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-VEC4-SVE:       %[[vscale:.*]] = vector.vscale
// CHECK-VEC4-SVE:       %[[step:.*]] = arith.muli %[[vscale]], %[[c4]] : index
// CHECK-VEC4-SVE:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[step]] {
// CHECK-VEC4-SVE:         %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[i]])[%[[step]]]
// CHECK-VEC4-SVE:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<[4]xi1>
// CHECK-VEC4-SVE:         %[[li:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %[[v0i]] : memref<?xi32>, vector<[4]xi1>, vector<[4]xi32> into vector<[4]xi32>
// CHECK-VEC4-SVE:         %[[lii64:.*]] = arith.extui %[[li]] : vector<[4]xi32> to vector<[4]xi64>
// CHECK-VEC4-SVE:         %[[la:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %[[v0f]] : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[lb:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[lii64]]], %[[mask]], %[[v0f]] : memref<1024xf32>, vector<[4]xi64>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<[4]xf32>
// CHECK-VEC4-SVE:         vector.scatter %{{.*}}[%[[c0]]] [%[[lii64]]], %[[mask]], %[[m]] : memref<1024xf32>, vector<[4]xi64>, vector<[4]xi1>, vector<[4]xf32>
// CHECK-VEC4-SVE:       }
// CHECK-VEC4-SVE:       return
//
func.func @mul_s(%arga: tensor<1024xf32, #SparseVector>,
                 %argb: tensor<1024xf32>,
		 %argx: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = linalg.generic #trait_mul_s
    ins(%arga, %argb: tensor<1024xf32, #SparseVector>, tensor<1024xf32>)
    outs(%argx: tensor<1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// -----

#DenseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>

#trait_reduction_d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"],
  doc = "x += a(i) * b(i)"
}

//
// CHECK-SCALAR-LABEL: func @reduction_d
// CHECK-SCALAR-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-SCALAR-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-SCALAR-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-SCALAR:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] iter_args(%[[red_in:.*]] = %{{.*}}) -> (f32) {
// CHECK-SCALAR:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-SCALAR:         %[[lb:.*]] = memref.load %{{.*}}[%[[i]]] : memref<1024xf32>
// CHECK-SCALAR:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : f32
// CHECK-SCALAR:         %[[a:.*]] = arith.addf %[[red_in]], %[[m]] : f32
// CHECK-SCALAR:         scf.yield %[[a]] : f32
// CHECK-SCALAR:       }
// CHECK-SCALAR:       return
//
// CHECK-VEC16-LABEL: func @reduction_d
// CHECK-VEC16-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-VEC16-DAG:   %[[v0:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-VEC16:       %[[l:.*]] = memref.load %{{.*}}[] : memref<f32>
// CHECK-VEC16:       %[[r:.*]] = vector.insert %[[l]], %[[v0]] [0] : f32 into vector<16xf32>
// CHECK-VEC16:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] iter_args(%[[red_in:.*]] = %[[r]]) -> (vector<16xf32>) {
// CHECK-VEC16:         %[[la:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC16:         %[[lb:.*]] = vector.load %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC16:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC16:         %[[a:.*]] = arith.addf %[[red_in]], %[[m]] : vector<16xf32>
// CHECK-VEC16:         scf.yield %[[a]] : vector<16xf32>
// CHECK-VEC16:       }
// CHECK-VEC16:       %{{.*}} = vector.reduction <add>, %[[red]] : vector<16xf32> into f32
// CHECK-VEC16:       return
//
// CHECK-VEC16-IDX32-LABEL: func @reduction_d
// CHECK-VEC16-IDX32-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-IDX32-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-IDX32-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-VEC16-IDX32-DAG:   %[[v0:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-VEC16-IDX32:       %[[l:.*]] = memref.load %{{.*}}[] : memref<f32>
// CHECK-VEC16-IDX32:       %[[r:.*]] = vector.insert %[[l]], %[[v0]] [0] : f32 into vector<16xf32>
// CHECK-VEC16-IDX32:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] iter_args(%[[red_in:.*]] = %[[r]]) -> (vector<16xf32>) {
// CHECK-VEC16-IDX32:         %[[la:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC16-IDX32:         %[[lb:.*]] = vector.load %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC16-IDX32:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC16-IDX32:         %[[a:.*]] = arith.addf %[[red_in]], %[[m]] : vector<16xf32>
// CHECK-VEC16-IDX32:         scf.yield %[[a]] : vector<16xf32>
// CHECK-VEC16-IDX32:       }
// CHECK-VEC16-IDX32:       %{{.*}} = vector.reduction <add>, %[[red]] : vector<16xf32> into f32
// CHECK-VEC16-IDX32:       return
//
// CHECK-VEC4-SVE:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)
// CHECK-VEC4-SVE-LABEL: func @reduction_d
// CHECK-VEC4-SVE-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC4-SVE-DAG:   %[[c4:.*]] = arith.constant 4 : index
// CHECK-VEC4-SVE-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-VEC4-SVE-DAG:   %[[v0:.*]] = arith.constant dense<0.000000e+00> : vector<[4]xf32>
// CHECK-VEC4-SVE:       %[[l:.*]] = memref.load %{{.*}}[] : memref<f32>
// CHECK-VEC4-SVE:       %[[vscale:.*]] = vector.vscale
// CHECK-VEC4-SVE:       %[[step:.*]] = arith.muli %[[vscale]], %[[c4]] : index
// CHECK-VEC4-SVE:       %[[r:.*]] = vector.insert %[[l]], %[[v0]] [0] : f32 into vector<[4]xf32>
// CHECK-VEC4-SVE:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[step]] iter_args(%[[red_in:.*]] = %[[r]]) -> (vector<[4]xf32>) {
// CHECK-VEC4-SVE:         %[[sub:.*]] = affine.min #[[$map]](%[[c1024]], %[[i]])[%[[step]]]
// CHECK-VEC4-SVE:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<[4]xi1>
// CHECK-VEC4-SVE:         %[[la:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %[[v0]] : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[lb:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %[[v0]] : memref<1024xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[a:.*]] = arith.addf %[[red_in]], %[[m]] : vector<[4]xf32>
// CHECK-VEC4-SVE:         %[[sa:.*]] = arith.select %[[mask]], %[[a]], %[[red_in]] : vector<[4]xi1>, vector<[4]xf32>
// CHECK-VEC4-SVE:         scf.yield %[[sa]] : vector<[4]xf32>
// CHECK-VEC4-SVE:       }
// CHECK-VEC4-SVE:       %{{.*}} = vector.reduction <add>, %[[red]] : vector<[4]xf32> into f32
// CHECK-VEC4-SVE:       return
//
func.func @reduction_d(%arga: tensor<1024xf32, #DenseVector>,
                       %argb: tensor<1024xf32>,
		       %argx: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic #trait_reduction_d
    ins(%arga, %argb: tensor<1024xf32, #DenseVector>, tensor<1024xf32>)
    outs(%argx: tensor<f32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %x, %0 : f32
        linalg.yield %1 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#trait_mul_ds = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * B(i,j)"
}

//
// CHECK-SCALAR-LABEL: func @mul_ds
// CHECK-SCALAR-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-SCALAR-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-SCALAR-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK-SCALAR:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-SCALAR:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-SCALAR:         %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-SCALAR:         %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-SCALAR:         %[[a:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-SCALAR:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-SCALAR:         %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-SCALAR:         %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-SCALAR:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK-SCALAR:           %[[lj:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xi32>
// CHECK-SCALAR:           %[[zj:.*]] = arith.extui %[[lj]] : i32 to i64
// CHECK-SCALAR:           %[[cj:.*]] = arith.index_cast %[[zj]] : i64 to index
// CHECK-SCALAR:           %[[la:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xf32>
// CHECK-SCALAR:           %[[lb:.*]] = memref.load %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK-SCALAR:           %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : f32
// CHECK-SCALAR:           store %[[m]], %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK-SCALAR:         }
// CHECK-SCALAR:       }
// CHECK-SCALAR:       return
//
// CHECK-VEC16:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC16-LABEL: func @mul_ds
// CHECK-VEC16-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC16-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK-VEC16:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC16:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC16:         %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-VEC16:         %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-VEC16:         %[[a:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-VEC16:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC16:         %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-VEC16:         %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-VEC16:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC16:           %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[j]])[%[[c16]]]
// CHECK-VEC16:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC16:           %[[lj:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC16:           %[[zj:.*]] = arith.extui %[[lj]] : vector<16xi32> to vector<16xi64>
// CHECK-VEC16:           %[[la:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16:           %[[lb:.*]] = vector.gather %{{.*}}[%[[i]], %[[c0]]] [%[[zj]]], %[[mask]], %{{.*}} : memref<512x1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16:           %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC16:           vector.scatter %{{.*}}[%[[i]], %[[c0]]] [%[[zj]]], %[[mask]], %[[m]] : memref<512x1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32>
// CHECK-VEC16:         }
// CHECK-VEC16:       }
// CHECK-VEC16:       return
//
// CHECK-VEC16-IDX32:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC16-IDX32-LABEL: func @mul_ds
// CHECK-VEC16-IDX32-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-IDX32-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC16-IDX32-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-IDX32-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK-VEC16-IDX32:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC16-IDX32:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC16-IDX32:         %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-VEC16-IDX32:         %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-VEC16-IDX32:         %[[a:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-VEC16-IDX32:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC16-IDX32:         %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-VEC16-IDX32:         %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-VEC16-IDX32:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC16-IDX32:           %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[j]])[%[[c16]]]
// CHECK-VEC16-IDX32:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC16-IDX32:           %[[lj:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC16-IDX32:           %[[la:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16-IDX32:           %[[lb:.*]] = vector.gather %{{.*}}[%[[i]], %[[c0]]] [%[[lj]]], %[[mask]], %{{.*}} : memref<512x1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC16-IDX32:           %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC16-IDX32:           vector.scatter %{{.*}}[%[[i]], %[[c0]]] [%[[lj]]], %[[mask]], %[[m]] : memref<512x1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
// CHECK-VEC16-IDX32:         }
// CHECK-VEC16-IDX32:       }
// CHECK-VEC16-IDX32:       return
//
// CHECK-VEC4-SVE:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)
// CHECK-VEC4-SVE-LABEL: func @mul_ds
// CHECK-VEC4-SVE-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC4-SVE-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC4-SVE-DAG:   %[[c4:.*]] = arith.constant 4 : index
// CHECK-VEC4-SVE-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK-VEC4-SVE-DAG:   %[[v0i:.*]] = arith.constant dense<0> : vector<[4]xi32>
// CHECK-VEC4-SVE-DAG:   %[[v0f:.*]] = arith.constant dense<0.000000e+00> : vector<[4]xf32>
// CHECK-VEC4-SVE:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC4-SVE:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC4-SVE:         %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK-VEC4-SVE:         %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK-VEC4-SVE:         %[[a:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-VEC4-SVE:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC4-SVE:         %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK-VEC4-SVE:         %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK-VEC4-SVE:         %[[vscale:.*]] = vector.vscale
// CHECK-VEC4-SVE:         %[[step:.*]] = arith.muli %[[vscale]], %[[c4]] : index
// CHECK-VEC4-SVE:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[step]] {
// CHECK-VEC4-SVE:           %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[j]])[%[[step]]]
// CHECK-VEC4-SVE:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<[4]xi1>
// CHECK-VEC4-SVE:           %[[lji32:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %[[v0i]] : memref<?xi32>, vector<[4]xi1>, vector<[4]xi32> into vector<[4]xi32>
// CHECK-VEC4-SVE:           %[[lj:.*]] = arith.extui %[[lji32]] : vector<[4]xi32> to vector<[4]xi64>
// CHECK-VEC4-SVE:           %[[la:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %[[v0f]] : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:           %[[lb:.*]] = vector.gather %{{.*}}[%[[i]], %[[c0]]] [%[[lj]]], %[[mask]], %[[v0f]] : memref<512x1024xf32>, vector<[4]xi64>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
// CHECK-VEC4-SVE:           %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : vector<[4]xf32>
// CHECK-VEC4-SVE:           vector.scatter %{{.*}}[%[[i]], %[[c0]]] [%[[lj]]], %[[mask]], %[[m]] : memref<512x1024xf32>, vector<[4]xi64>, vector<[4]xi1>, vector<[4]xf32>
// CHECK-VEC4-SVE:         }
// CHECK-VEC4-SVE:       }
// CHECK-VEC4-SVE:       return
//
func.func @mul_ds(%arga: tensor<512x1024xf32, #SparseMatrix>,
                  %argb: tensor<512x1024xf32>,
		  %argx: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
  %0 = linalg.generic #trait_mul_ds
    ins(%arga, %argb: tensor<512x1024xf32, #SparseMatrix>, tensor<512x1024xf32>)
    outs(%argx: tensor<512x1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<512x1024xf32>
  return %0 : tensor<512x1024xf32>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>

#trait_affine = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i+1,j)>
  ],
  iterator_types = ["parallel","parallel"],
  doc = "X(i+1,j) += A(i,j)"
}

//
// CHECK-SCALAR-LABEL: func @add_dense
// CHECK-SCALAR-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-SCALAR-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-SCALAR-DAG:   %[[c32:.*]] = arith.constant 32 : index
// CHECK-SCALAR:       scf.for %[[i:.*]] = %[[c0]] to %[[c32]] step %[[c1]] {
// CHECK-SCALAR:         %[[lo:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xindex>
// CHECK-SCALAR:         %[[i1:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-SCALAR:         %[[hi:.*]] = memref.load %{{.*}}[%[[i1]]] : memref<?xindex>
// CHECK-SCALAR:         scf.for %[[jj:.*]] = %[[lo]] to %[[hi]] step %[[c1]] {
// CHECK-SCALAR:           %[[j:.*]] = memref.load %{{.*}}[%[[jj]]] : memref<?xindex>
// CHECK-SCALAR:           %[[x:.*]] = memref.load %{{.*}}[%[[i1]], %[[j]]] : memref<33x64xf64>
// CHECK-SCALAR:           %[[a:.*]] = memref.load %{{.*}}[%[[jj]]] : memref<?xf64>
// CHECK-SCALAR:           %[[s:.*]] = arith.addf %[[x]], %[[a]] : f64
// CHECK-SCALAR:           memref.store %[[s]], %{{.*}}[%[[i1]], %[[j]]] : memref<33x64xf64>
// CHECK-SCALAR:         }
// CHECK-SCALAR:       }
// CHECK-SCALAR:       return
//
// CHECK-VEC16:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC16-LABEL: func @add_dense
// CHECK-VEC16-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC16-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-DAG:   %[[c32:.*]] = arith.constant 32 : index
// CHECK-VEC16:       scf.for %[[i:.*]] = %[[c0]] to %[[c32]] step %[[c1]] {
// CHECK-VEC16:         %[[lo:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xindex>
// CHECK-VEC16:         %[[i1:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-VEC16:         %[[hi:.*]] = memref.load %{{.*}}[%[[i1]]] : memref<?xindex>
// CHECK-VEC16:         scf.for %[[jj:.*]] = %[[lo]] to %[[hi]] step %[[c16]] {
// CHECK-VEC16:           %[[sub:.*]] = affine.min #[[$map]](%[[hi]], %[[jj]])[%[[c16]]]
// CHECK-VEC16:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC16:           %[[j:.*]] = vector.maskedload %{{.*}}[%[[jj]]], %[[mask]], %{{.*}} : memref<?xindex>
// CHECK-VEC16:           %[[x:.*]] = vector.gather %{{.*}}[%[[i1]], %[[c0]]] [%[[j]]], %[[mask]], %{{.*}} : memref<33x64xf64>
// CHECK-VEC16:           %[[a:.*]] = vector.maskedload %{{.*}}[%[[jj]]], %[[mask]], %{{.*}} : memref<?xf64>
// CHECK-VEC16:           %[[s:.*]] = arith.addf %[[x]], %[[a]] : vector<16xf64>
// CHECK-VEC16:           vector.scatter %{{.*}}[%[[i1]], %[[c0]]] [%[[j]]], %[[mask]], %[[s]] : memref<33x64xf64>
// CHECK-VEC16:         }
// CHECK-VEC16:       }
// CHECK-VEC16:       return
//
// CHECK-VEC16-IDX32:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC16-IDX32-LABEL: func @add_dense
// CHECK-VEC16-IDX32-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC16-IDX32-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC16-IDX32-DAG:   %[[c16:.*]] = arith.constant 16 : index
// CHECK-VEC16-IDX32-DAG:   %[[c32:.*]] = arith.constant 32 : index
// CHECK-VEC16-IDX32:       scf.for %[[i:.*]] = %[[c0]] to %[[c32]] step %[[c1]] {
// CHECK-VEC16-IDX32:         %[[lo:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xindex>
// CHECK-VEC16-IDX32:         %[[i1:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-VEC16-IDX32:         %[[hi:.*]] = memref.load %{{.*}}[%[[i1]]] : memref<?xindex>
// CHECK-VEC16-IDX32:         scf.for %[[jj:.*]] = %[[lo]] to %[[hi]] step %[[c16]] {
// CHECK-VEC16-IDX32:           %[[sub:.*]] = affine.min #[[$map]](%[[hi]], %[[jj]])[%[[c16]]]
// CHECK-VEC16-IDX32:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC16-IDX32:           %[[j:.*]] = vector.maskedload %{{.*}}[%[[jj]]], %[[mask]], %{{.*}} : memref<?xindex>
// CHECK-VEC16-IDX32:           %[[x:.*]] = vector.gather %{{.*}}[%[[i1]], %[[c0]]] [%[[j]]], %[[mask]], %{{.*}} : memref<33x64xf64>
// CHECK-VEC16-IDX32:           %[[a:.*]] = vector.maskedload %{{.*}}[%[[jj]]], %[[mask]], %{{.*}} : memref<?xf64>
// CHECK-VEC16-IDX32:           %[[s:.*]] = arith.addf %[[x]], %[[a]] : vector<16xf64>
// CHECK-VEC16-IDX32:           vector.scatter %{{.*}}[%[[i1]], %[[c0]]] [%[[j]]], %[[mask]], %[[s]] : memref<33x64xf64>
// CHECK-VEC16-IDX32:         }
// CHECK-VEC16-IDX32:       }
// CHECK-VEC16-IDX32:       return
//
// CHECK-VEC4-SVE:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)
// CHECK-VEC4-SVE-LABEL: func @add_dense
// CHECK-VEC4-SVE-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-VEC4-SVE-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-VEC4-SVE-DAG:   %[[c4:.*]] = arith.constant 4 : index
// CHECK-VEC4-SVE-DAG:   %[[c32:.*]] = arith.constant 32 : index
// CHECK-VEC4-SVE-DAG:   %[[v0idx:.*]] = arith.constant dense<0> : vector<[4]xindex>
// CHECK-VEC4-SVE-DAG:   %[[v0f64:.*]] = arith.constant dense<0.000000e+00> : vector<[4]xf64>
// CHECK-VEC4-SVE:       scf.for %[[i:.*]] = %[[c0]] to %[[c32]] step %[[c1]] {
// CHECK-VEC4-SVE:         %[[lo:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xindex>
// CHECK-VEC4-SVE:         %[[i1:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK-VEC4-SVE:         %[[hi:.*]] = memref.load %{{.*}}[%[[i1]]] : memref<?xindex>
// CHECK-VEC4-SVE:         %[[vscale:.*]] = vector.vscale
// CHECK-VEC4-SVE:         %[[step:.*]] = arith.muli %[[vscale]], %[[c4]] : index
// CHECK-VEC4-SVE:         scf.for %[[jj:.*]] = %[[lo]] to %[[hi]] step %[[step]] {
// CHECK-VEC4-SVE:           %[[sub:.*]] = affine.min #[[$map]](%[[hi]], %[[jj]])[%[[step]]]
// CHECK-VEC4-SVE:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<[4]xi1>
// CHECK-VEC4-SVE:           %[[j:.*]] = vector.maskedload %{{.*}}[%[[jj]]], %[[mask]], %[[v0idx]] : memref<?xindex>
// CHECK-VEC4-SVE:           %[[x:.*]] = vector.gather %{{.*}}[%[[i1]], %[[c0]]] [%[[j]]], %[[mask]], %[[v0f64]] : memref<33x64xf64>
// CHECK-VEC4-SVE:           %[[a:.*]] = vector.maskedload %{{.*}}[%[[jj]]], %[[mask]], %[[v0f64]] : memref<?xf64>
// CHECK-VEC4-SVE:           %[[s:.*]] = arith.addf %[[x]], %[[a]] : vector<[4]xf64>
// CHECK-VEC4-SVE:           vector.scatter %{{.*}}[%[[i1]], %[[c0]]] [%[[j]]], %[[mask]], %[[s]] : memref<33x64xf64>
// CHECK-VEC4-SVE:         }
// CHECK-VEC4-SVE:       }
// CHECK-VEC4-SVE:       return
//
func.func @add_dense(%arga: tensor<32x64xf64, #SparseMatrix>,
                     %argx: tensor<33x64xf64>) -> tensor<33x64xf64> {
  %0 = linalg.generic #trait_affine
     ins(%arga: tensor<32x64xf64, #SparseMatrix>)
    outs(%argx: tensor<33x64xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = arith.addf %x, %a : f64
        linalg.yield %0 : f64
  } -> tensor<33x64xf64>
  return %0 : tensor<33x64xf64>
}
