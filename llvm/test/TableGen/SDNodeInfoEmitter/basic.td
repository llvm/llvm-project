// RUN: split-file %s %t

//--- no-nodes.td
// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %t/no-nodes.td \
// RUN:   | FileCheck %t/no-nodes.td

include "llvm/Target/Target.td"

def MyTarget : Target;

// CHECK:       #ifdef GET_SDNODE_ENUM
// CHECK-NEXT:  #undef GET_SDNODE_ENUM
// CHECK-EMPTY:
// CHECK-NEXT:  namespace llvm::MyTargetISD {
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr unsigned GENERATED_OPCODE_END = ISD::BUILTIN_OP_END;
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace llvm::MyTargetISD
// CHECK-EMPTY:
// CHECK-NEXT:  #endif // GET_SDNODE_ENUM
// CHECK-EMPTY:
// CHECK-NEXT:  #ifdef GET_SDNODE_DESC
// CHECK-NEXT:  #undef GET_SDNODE_DESC
// CHECK-EMPTY:
// CHECK-NEXT:  namespace llvm {
// CHECK-EMPTY:
// CHECK-NEXT:  #ifdef __GNUC__
// CHECK-NEXT:  #pragma GCC diagnostic push
// CHECK-NEXT:  #pragma GCC diagnostic ignored "-Woverlength-strings"
// CHECK-NEXT:  #endif
// CHECK-NEXT:  static const char MyTargetSDNodeNames[] =
// CHECK-NEXT:    "\0";
// CHECK-NEXT:  #ifdef __GNUC__
// CHECK-NEXT:  #pragma GCC diagnostic pop
// CHECK-NEXT:  #endif
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* dummy */ {SDTCisVT, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/0, MyTargetSDNodeDescs,
// CHECK-NEXT:      MyTargetSDNodeNames, MyTargetSDTypeConstraints);
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace llvm


//--- trivial-node.td
// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %t/trivial-node.td \
// RUN:   | FileCheck %t/trivial-node.td

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

// CHECK:       static const char MyTargetSDNodeNames[] =
// CHECK-NEXT:    "MyTargetISD::NOOP\0"
// CHECK-NEXT:    "\0";

// CHECK:       static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* dummy */ {SDTCisVT, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:      {0, 0, 0, 0, 0, 0, 0, 0}, // NOOP
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs,
// CHECK-NEXT:      MyTargetSDNodeNames, MyTargetSDTypeConstraints);
