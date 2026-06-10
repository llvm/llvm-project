// The following test examples of linalg category ops lowered to linalg.generic
// and then lifted back up to category op.

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=category-to-generic \
// RUN: | mlir-opt -split-input-file -linalg-morph-ops=generic-to-category \
// RUN: | FileCheck %s

func.func @unary_ops(%A: memref<7x14x21xf32>, %Out: memref<7x14x21xf32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<exp>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<log>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<abs>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<ceil>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<floor>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<negf>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<reciprocal>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<round>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<sqrt>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<square>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<tanh>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<erf>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<sin>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<cos>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<tan>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<acos>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<acosh>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<asin>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<asinh>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<atan>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<atanh>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<log10>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<log1p>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<log2>
    ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  return
}

// CHECK-LABEL: unary_ops
// CHECK-SAME: %[[A:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<abs>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<ceil>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<floor>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<negf>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<reciprocal>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<round>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sqrt>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<square>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<tanh>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<erf>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sin>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<cos>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<tan>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<acos>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<acosh>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<asin>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<asinh>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<atan>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<atanh>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log10>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log1p>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log2>
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// -----

func.func @unary_ops_non_identity(%A: tensor<?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise
    kind=#linalg.elementwise_kind<log>
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>]
    ins(%A : tensor<?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[MAP_BC:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP_TP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: unary_ops_non_identity
// CHECK-SAME: %[[A:.+]]: tensor<?xf32>, %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<log>
// CHECK-SAME: indexing_maps = [#[[MAP_BC]], #[[MAP_TP]]]
// CHECK-SAME: ins(%[[A]] : tensor<?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----
              
func.func @binary_ops_int(%A: memref<10xi32>, %B: memref<10xi32>,
                          %Out: memref<10xi32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<sub>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<div>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<div_unsigned>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<min_signed>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  return
}

// CHECK-LABEL: binary_ops_int
// CHECK-SAME: %[[A:.+]]: memref<10xi32>, %[[B:.+]]: memref<10xi32>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xi32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sub>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<div>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<div_unsigned>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<min_signed>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)

// -----
              
func.func @binary_ops_float(%A: memref<10xf32>, %B: memref<10xf32>,
                            %Out: memref<10xf32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<sub>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<div>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<min_signed>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.elementwise kind=#linalg.elementwise_kind<powf>
    ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  return
}

// CHECK-LABEL: binary_ops_float
// CHECK-SAME: %[[A:.+]]: memref<10xf32>, %[[B:.+]]: memref<10xf32>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sub>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<div>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<min_signed>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<powf>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)

// -----
              
func.func @binary_ops_complex(%A: memref<10xcomplex<f32>>, %B: memref<10xcomplex<f32>>,
                              %Out: memref<10xcomplex<f32>>) {
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
    outs(%Out : memref<10xcomplex<f32>>)
  linalg.elementwise kind=#linalg.elementwise_kind<sub>
    ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
    outs(%Out : memref<10xcomplex<f32>>)
  linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
    outs(%Out : memref<10xcomplex<f32>>)
  linalg.elementwise kind=#linalg.elementwise_kind<div>
    ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
    outs(%Out : memref<10xcomplex<f32>>)
  return
}

// CHECK-LABEL: binary_ops_complex
// CHECK-SAME: %[[A:.+]]: memref<10xcomplex<f32>>, %[[B:.+]]: memref<10xcomplex<f32>>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xcomplex<f32>>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sub>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<div>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)

// -----
              
func.func @binary_ops_bool(%A: memref<10xi1>, %B: memref<10xi1>,
                           %Out: memref<10xi1>) {
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%A, %B : memref<10xi1>, memref<10xi1>) outs(%Out : memref<10xi1>)
  linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%A, %B : memref<10xi1>, memref<10xi1>) outs(%Out : memref<10xi1>)
  return
}

// CHECK-LABEL: binary_ops_bool
// CHECK-SAME: %[[A:.+]]: memref<10xi1>, %[[B:.+]]: memref<10xi1>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xi1>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi1>, memref<10xi1>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi1>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi1>, memref<10xi1>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi1>)

// -----
              
func.func @binary_ops_uint(%A: memref<10xi32>, %B: memref<10xi32>,
                           %Out: memref<10xi32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<max_unsigned>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.elementwise kind=#linalg.elementwise_kind<min_unsigned>
    ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  return
}

// CHECK-LABEL: binary_ops_uint
// CHECK-SAME: %[[A:.+]]: memref<10xi32>, %[[B:.+]]: memref<10xi32>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xi32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<max_unsigned>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<min_unsigned>
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)

// -----

func.func @binary_ops_non_identity(%A: tensor<?xf32>, %B: tensor<?x?xf32>,
                                   %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise
    kind=#linalg.elementwise_kind<add>
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>]
    ins(%A, %B : tensor<?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[MAP_BC:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP_TP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP_ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: binary_ops_non_identity
// CHECK-SAME: %[[A:.+]]: tensor<?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: indexing_maps = [#[[MAP_BC]], #[[MAP_TP]], #[[MAP_ID]]]
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @contract_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contract_matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @contract_matmul_bool(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi1>,
    %arg2: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<?x?xi1>, tensor<?x?xi1>)
    outs(%arg2 : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// CHECK-LABEL: contract_matmul_bool
// CHECK-SAME: %[[A:.+]]: tensor<?x?xi1>, %[[B:.+]]: tensor<?x?xi1>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xi1>, tensor<?x?xi1>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi1>) -> tensor<?x?xi1>

func.func @contract_matmul_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
    %arg2: memref<?x?xf32>) {
  linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
    outs(%arg2 : memref<?x?xf32>)
  return
}

// CHECK-LABEL: contract_matmul_memref
// CHECK-SAME: %[[A:.+]]: memref<?x?xf32>, %[[B:.+]]: memref<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: memref<?x?xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<?x?xf32>)

func.func @contract_matmul_bitcast_int_to_float(%arg0: tensor<16x8xi32>,
    %arg1: tensor<8x32xi32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<16x8xi32>, tensor<8x32xi32>)
    outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: contract_matmul_bitcast_int_to_float
// CHECK-SAME: %[[A:.+]]: tensor<16x8xi32>, %[[B:.+]]: tensor<8x32xi32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-NOT: cast =
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<16x8xi32>, tensor<8x32xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<16x32xf32>) -> tensor<16x32xf32>

func.func @contract_matmul_unsigned_cast_float(%arg0: tensor<16x8xi16>,
    %arg1: tensor<8x32xi16>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    {cast = #linalg.type_fn<cast_unsigned>}
    ins(%arg0, %arg1 : tensor<16x8xi16>, tensor<8x32xi16>)
    outs(%arg2 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: contract_matmul_unsigned_cast_float
// CHECK-SAME: %[[A:.+]]: tensor<16x8xi16>, %[[B:.+]]: tensor<8x32xi16>,
// CHECK-SAME: %[[OUT:.+]]: tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
// CHECK-SAME: cast = #linalg.type_fn<cast_unsigned>
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<16x8xi16>, tensor<8x32xi16>)
// CHECK-SAME: outs(%[[OUT]] : tensor<16x32xf32>) -> tensor<16x32xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @contract_multi_reduction(%arg0: tensor<10x20x30xf32>,
    %arg1: tensor<30x20x40xf32>, %arg2: tensor<10x40xf32>) -> tensor<10x40xf32> {
  %0 = linalg.contract indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
    outs(%arg2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %0 : tensor<10x40xf32>
}

// CHECK-DAG: #[[$MAP_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$MAP_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
// CHECK-DAG: #[[$MAP_C:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: contract_multi_reduction
// CHECK-NOT: linalg.generic
// CHECK: linalg.contract
// CHECK-SAME: indexing_maps = {{\[}}#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]{{\]}}
