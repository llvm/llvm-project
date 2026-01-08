// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

def my_node_a : SDNode<"MyTargetISD::NODE", SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>>;
def my_node_b : SDNode<"MyTargetISD::NODE", SDTypeProfile<1, 0, [SDTCisVT<0, f32>]>>;

// CHECK:       enum GenNodeType : unsigned {
// CHECK-NEXT:    NODE = ISD::BUILTIN_OP_END,
// CHECK-NEXT:  };

// CHECK:       static constexpr char MyTargetSDNodeNamesStorage[] =
// CHECK-NEXT:    "\0"
// CHECK-NEXT:    "MyTargetISD::NODE\0"
// CHECK-NEXT:    ;

// CHECK:       static const VTByHwModePair MyTargetVTByHwModeTable[] = {
// CHECK-NEXT:   /* dummy */ {0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* dummy */ {SDTCisVT, 0, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:      {1, 0, 0, 0, 0, 1, 0, 0}, // NODE
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// CHECK-NEXT:      MyTargetVTByHwModeTable, MyTargetSDTypeConstraints);
