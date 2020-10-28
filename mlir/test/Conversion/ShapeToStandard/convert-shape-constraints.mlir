// RUN: mlir-opt -convert-shape-constraints <%s | FileCheck %s

// There's not very much useful to check here other than pasting the output.
// CHECK-LABEL:   func @cstr_broadcastable(
// CHECK-SAME:                             %[[LHS:.*]]: tensor<?xindex>,
// CHECK-SAME:                             %[[RHS:.*]]: tensor<?xindex>) -> !shape.witness {
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[RET:.*]] = shape.const_witness true
// CHECK:           %[[LHS_RANK:.*]] = dim %[[LHS]], %[[C0]] : tensor<?xindex>
// CHECK:           %[[RHS_RANK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?xindex>
// CHECK:           %[[LHS_RANK_ULE:.*]] = cmpi "ule", %[[LHS_RANK]], %[[RHS_RANK]] : index
// CHECK:           %[[LESSER_RANK:.*]] = select %[[LHS_RANK_ULE]], %[[LHS_RANK]], %[[RHS_RANK]] : index
// CHECK:           %[[GREATER_RANK:.*]] = select %[[LHS_RANK_ULE]], %[[RHS_RANK]], %[[LHS_RANK]] : index
// CHECK:           %[[LESSER_RANK_OPERAND:.*]] = select %[[LHS_RANK_ULE]], %[[LHS]], %[[RHS]] : tensor<?xindex>
// CHECK:           %[[GREATER_RANK_OPERAND:.*]] = select %[[LHS_RANK_ULE]], %[[RHS]], %[[LHS]] : tensor<?xindex>
// CHECK:           %[[RANK_DIFF:.*]] = subi %[[GREATER_RANK]], %[[LESSER_RANK]] : index
// CHECK:           scf.for %[[IV:.*]] = %[[RANK_DIFF]] to %[[GREATER_RANK]] step %[[C1]] {
// CHECK:             %[[GREATER_RANK_OPERAND_EXTENT:.*]] = extract_element %[[GREATER_RANK_OPERAND]][%[[IV]]] : tensor<?xindex>
// CHECK:             %[[IVSHIFTED:.*]] = subi %[[IV]], %[[RANK_DIFF]] : index
// CHECK:             %[[LESSER_RANK_OPERAND_EXTENT:.*]] = extract_element %[[LESSER_RANK_OPERAND]][%[[IVSHIFTED]]] : tensor<?xindex>
// CHECK:             %[[GREATER_RANK_OPERAND_EXTENT_IS_ONE:.*]] = cmpi "eq", %[[GREATER_RANK_OPERAND_EXTENT]], %[[C1]] : index
// CHECK:             %[[LESSER_RANK_OPERAND_EXTENT_IS_ONE:.*]] = cmpi "eq", %[[LESSER_RANK_OPERAND_EXTENT]], %[[C1]] : index
// CHECK:             %[[EXTENTS_AGREE:.*]] = cmpi "eq", %[[GREATER_RANK_OPERAND_EXTENT]], %[[LESSER_RANK_OPERAND_EXTENT]] : index
// CHECK:             %[[OR_TMP:.*]] = or %[[GREATER_RANK_OPERAND_EXTENT_IS_ONE]], %[[LESSER_RANK_OPERAND_EXTENT_IS_ONE]] : i1
// CHECK:             %[[BROADCAST_IS_VALID:.*]] = or %[[EXTENTS_AGREE]], %[[OR_TMP]] : i1
// CHECK:             assert %[[BROADCAST_IS_VALID]], "invalid broadcast"
// CHECK:           }
// CHECK:           return %[[RET]] : !shape.witness
// CHECK:         }
func @cstr_broadcastable(%arg0: tensor<?xindex>, %arg1: tensor<?xindex>) -> !shape.witness {
  %witness = shape.cstr_broadcastable %arg0, %arg1 : tensor<?xindex>, tensor<?xindex>
  return %witness : !shape.witness
}

// CHECK-LABEL: func @cstr_require
func @cstr_require(%arg0: i1) -> !shape.witness {
  // CHECK: %[[RET:.*]] = shape.const_witness true
  // CHECK: assert %arg0, "msg"
  // CHECK: return %[[RET]]
  %witness = shape.cstr_require %arg0, "msg"
  return %witness : !shape.witness
}
