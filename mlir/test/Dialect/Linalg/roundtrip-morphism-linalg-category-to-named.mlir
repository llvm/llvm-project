// The following test examples of linalg named/category ops lowered to the other
// form and then lifted back up.

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=category-to-named \
// RUN: | mlir-opt -split-input-file -linalg-morph-ops=named-to-category \
// RUN: | FileCheck %s --check-prefix=CATEGORY

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=named-to-category \
// RUN: | mlir-opt -split-input-file -linalg-morph-ops=category-to-named \
// RUN: | FileCheck %s --check-prefix=NAMED

func.func @elementwise_exp(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.elementwise kind = #linalg.elementwise_kind<exp>
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CATEGORY-LABEL: @elementwise_exp
// CATEGORY-NOT: linalg.exp
// CATEGORY: linalg.elementwise
// CATEGORY-SAME: kind=#linalg.elementwise_kind<exp>

// NAMED-LABEL: @elementwise_exp
// NAMED-NOT: linalg.elementwise
// NAMED: linalg.exp

// -----

func.func @named_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.add
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CATEGORY-LABEL: @named_add
// CATEGORY-NOT: linalg.add
// CATEGORY: linalg.elementwise
// CATEGORY-SAME: kind=#linalg.elementwise_kind<add>

// NAMED-LABEL: @named_add
// NAMED-NOT: linalg.elementwise
// NAMED: linalg.add
