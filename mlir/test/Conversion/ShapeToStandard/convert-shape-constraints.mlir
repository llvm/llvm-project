// RUN: mlir-opt -convert-shape-constraints <%s | FileCheck %s

// There's not very much useful to check here other than pasting the output.
// CHECK-LABEL:   func @cstr_broadcastable(
// CHECK-SAME:                             %[[LHS:.*]]: tensor<?xindex>,
// CHECK-SAME:                             %[[RHS:.*]]: tensor<?xindex>) -> !shape.witness {
// CHECK:           %[[RET:.*]] = shape.const_witness true
// CHECK:           %[[BROADCAST_IS_VALID:.*]] = shape.is_broadcastable %[[LHS]], %[[RHS]]
// CHECK:           assert %[[BROADCAST_IS_VALID]], "required broadcastable shapes"
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
