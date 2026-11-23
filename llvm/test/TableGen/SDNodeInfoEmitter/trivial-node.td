// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

def my_noop : SDNode<"MyTargetISD::NOOP", SDTypeProfile<0, 0, []>>;

// CHECK:       namespace llvm::MyTargetISD {
// CHECK-EMPTY:
// CHECK-NEXT:  enum GenNodeType : unsigned {
// CHECK-NEXT:    NOOP = ISD::BUILTIN_OP_END,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr unsigned GENERATED_OPCODE_END = NOOP + 1;
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace llvm::MyTargetISD

// CHECK:       static constexpr char MyTargetSDNodeNamesStorage[] =
// CHECK-NEXT:    "\0"
// CHECK-NEXT:    "MyTargetISD::NOOP\0"
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
// CHECK-NEXT:      {0, 0, 0, 0, 0, 1, 0, 0}, // NOOP
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// CHECK-NEXT:      MyTargetVTByHwModeTable, MyTargetSDTypeConstraints);
