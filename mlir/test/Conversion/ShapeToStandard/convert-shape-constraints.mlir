// RUN: mlir-opt -convert-shape-constraints <%s | FileCheck %s

// There's not very much useful to check here other than pasting the output.
// CHECK-LABEL:   func @cstr_broadcastable(
// CHECK-SAME:                             %[[LHS:.*]]: tensor<?xindex>,
// CHECK-SAME:                             %[[RHS:.*]]: tensor<?xindex>) -> !shape.witness {
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[RET:.*]] = shape.const_witness true
// CHECK:           %[[LHSRANK:.*]] = dim %[[LHS]], %[[C0]] : tensor<?xindex>
// CHECK:           %[[RHSRANK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?xindex>
// CHECK:           %[[LESSEQUAL:.*]] = cmpi "ule", %[[LHSRANK]], %[[RHSRANK]] : index
// CHECK:           %[[IFRESULTS:.*]]:4 = scf.if %[[LESSEQUAL]] -> (index, tensor<?xindex>, index, tensor<?xindex>) {
// CHECK:             scf.yield %[[LHSRANK]], %[[LHS]], %[[RHSRANK]], %[[RHS]] : index, tensor<?xindex>, index, tensor<?xindex>
// CHECK:           } else {
// CHECK:             scf.yield %[[RHSRANK]], %[[RHS]], %[[LHSRANK]], %[[LHS]] : index, tensor<?xindex>, index, tensor<?xindex>
// CHECK:           }
// CHECK:           %[[RANKDIFF:.*]] = subi %[[IFRESULTS:.*]]#2, %[[IFRESULTS]]#0 : index
// CHECK:           scf.for %[[IV:.*]] = %[[RANKDIFF]] to %[[IFRESULTS]]#2 step %[[C1]] {
// CHECK:             %[[GREATERRANKOPERANDEXTENT:.*]] = extract_element %[[IFRESULTS]]#3{{\[}}%[[IV]]] : tensor<?xindex>
// CHECK:             %[[IVSHIFTED:.*]] = subi %[[IV]], %[[RANKDIFF]] : index
// CHECK:             %[[LESSERRANKOPERANDEXTENT:.*]] = extract_element %[[IFRESULTS]]#1{{\[}}%[[IVSHIFTED]]] : tensor<?xindex>
// CHECK:             %[[GREATERRANKOPERANDEXTENTISONE:.*]] = cmpi "eq", %[[GREATERRANKOPERANDEXTENT]], %[[C1]] : index
// CHECK:             %[[LESSERRANKOPERANDEXTENTISONE:.*]] = cmpi "eq", %[[LESSERRANKOPERANDEXTENT]], %[[C1]] : index
// CHECK:             %[[EXTENTSAGREE:.*]] = cmpi "eq", %[[GREATERRANKOPERANDEXTENT]], %[[LESSERRANKOPERANDEXTENT]] : index
// CHECK:             %[[OR_TMP:.*]] = or %[[GREATERRANKOPERANDEXTENTISONE]], %[[LESSERRANKOPERANDEXTENTISONE]] : i1
// CHECK:             %[[BROADCASTISVALID:.*]] = or %[[EXTENTSAGREE]], %[[OR_TMP]] : i1
// CHECK:             assert %[[BROADCASTISVALID]], "invalid broadcast"
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
