// RUN: mlir-opt -test-make-isolated-from-above=simple -allow-unregistered-dialect --split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-make-isolated-from-above=clone-ops-with-no-operands -allow-unregistered-dialect --split-input-file %s | FileCheck %s --check-prefix=CLONE1
// RUN: mlir-opt -test-make-isolated-from-above=clone-ops-with-operands -allow-unregistered-dialect --split-input-file %s | FileCheck %s --check-prefix=CLONE2

func.func @make_isolated_from_above_single_block(%arg0 : index, %arg1 : index) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %d0 = tensor.dim %empty, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %empty, %c1 : tensor<?x?xf32>
  "test.one_region_with_operands_op"() ({
    "foo.yield"(%c0, %c1, %d0, %d1) : (index, index, index, index) -> ()
  }) : () -> ()
  return
}
// CHECK-LABEL: func @make_isolated_from_above_single_block(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[ARG0]], %[[ARG1]])
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
//       CHECK:   test.isolated_one_region_op %[[C0]], %[[C1]], %[[D0]], %[[D1]]
//  CHECK-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index, %[[B3:[a-zA-Z0-9]+]]: index)
//       CHECK:       "foo.yield"(%[[B0]], %[[B1]], %[[B2]], %[[B3]])

// CLONE1-LABEL: func @make_isolated_from_above_single_block(
//  CLONE1-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CLONE1-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//   CLONE1-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CLONE1-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CLONE1-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[ARG0]], %[[ARG1]])
//   CLONE1-DAG:   %[[D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
//   CLONE1-DAG:   %[[D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
//       CLONE1:   test.isolated_one_region_op %[[D0]], %[[D1]]
//  CLONE1-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index)
//   CLONE1-DAG:       %[[C0_0:.+]] = arith.constant 0 : index
//   CLONE1-DAG:       %[[C1_0:.+]] = arith.constant 1 : index
//       CLONE1:       "foo.yield"(%[[C0_0]], %[[C1_0]], %[[B0]], %[[B1]])

// CLONE2-LABEL: func @make_isolated_from_above_single_block(
//  CLONE2-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CLONE2-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//       CLONE2:   test.isolated_one_region_op %[[ARG0]], %[[ARG1]]
//  CLONE2-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index)
//   CLONE2-DAG:       %[[C0:.+]] = arith.constant 0 : index
//   CLONE2-DAG:       %[[C1:.+]] = arith.constant 1 : index
//   CLONE2-DAG:       %[[EMPTY:.+]] = tensor.empty(%[[B0]], %[[B1]])
//   CLONE2-DAG:       %[[D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
//   CLONE2-DAG:       %[[D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
//       CLONE2:       "foo.yield"(%[[C0]], %[[C1]], %[[D0]], %[[D1]])

// -----

func.func @make_isolated_from_above_multiple_blocks(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %d0 = tensor.dim %empty, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %empty, %c1 : tensor<?x?xf32>
  "test.one_region_with_operands_op"(%arg2) ({
    ^bb0(%b0 : index):
      cf.br ^bb1(%b0: index)
    ^bb1(%b1 : index):
    "foo.yield"(%c0, %c1, %d0, %d1, %b1) : (index, index, index, index, index) -> ()
  }) : (index) -> ()
  return
}
// CHECK-LABEL: func @make_isolated_from_above_multiple_blocks(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[ARG0]], %[[ARG1]])
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
//       CHECK:   test.isolated_one_region_op %[[ARG2]], %[[C0]], %[[C1]], %[[D0]], %[[D1]]
//  CHECK-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index, %[[B3:[a-zA-Z0-9]+]]: index, %[[B4:[a-zA-Z0-9]+]]: index)
//  CHECK-NEXT:       cf.br ^bb1
//       CHECK:     ^bb1:
//       CHECK:       "foo.yield"(%[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B0]])

// CLONE1-LABEL: func @make_isolated_from_above_multiple_blocks(
//  CLONE1-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CLONE1-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CLONE1-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//   CLONE1-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CLONE1-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CLONE1-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[ARG0]], %[[ARG1]])
//   CLONE1-DAG:   %[[D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
//   CLONE1-DAG:   %[[D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
//       CLONE1:   test.isolated_one_region_op %[[ARG2]], %[[D0]], %[[D1]]
//  CLONE1-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
//   CLONE1-DAG:       %[[C0_0:.+]] = arith.constant 0 : index
//   CLONE1-DAG:       %[[C1_0:.+]] = arith.constant 1 : index
//  CLONE1-NEXT:       cf.br ^bb1
//       CLONE1:     ^bb1:
//       CLONE1:       "foo.yield"(%[[C0_0]], %[[C1_0]], %[[B1]], %[[B2]], %[[B0]])

// CLONE2-LABEL: func @make_isolated_from_above_multiple_blocks(
//  CLONE2-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CLONE2-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CLONE2-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//       CLONE2:   test.isolated_one_region_op %[[ARG2]], %[[ARG0]], %[[ARG1]]
//  CLONE2-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
//   CLONE2-DAG:       %[[C0:.+]] = arith.constant 0 : index
//   CLONE2-DAG:       %[[C1:.+]] = arith.constant 1 : index
//   CLONE2-DAG:       %[[EMPTY:.+]] = tensor.empty(%[[B1]], %[[B2]])
//   CLONE2-DAG:       %[[D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
//   CLONE2-DAG:       %[[D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
//  CLONE2-NEXT:       cf.br ^bb1
//       CLONE2:     ^bb1:
//       CLONE2:       "foo.yield"(%[[C0]], %[[C1]], %[[D0]], %[[D1]], %[[B0]])
