// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s | FileCheck %s

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
// CHECK-EMPTY:
// CHECK-NEXT:  #ifdef __GNUC__
// CHECK-NEXT:  #pragma GCC diagnostic push
// CHECK-NEXT:  #pragma GCC diagnostic ignored "-Woverlength-strings"
// CHECK-NEXT:  #endif
// CHECK-NEXT:  static constexpr char MyTargetSDNodeNamesStorage[] =
// CHECK-NEXT:    "\0"
// CHECK-NEXT:    ;
// CHECK-NEXT:  #ifdef __GNUC__
// CHECK-NEXT:  #pragma GCC diagnostic pop
// CHECK-NEXT:  #endif
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr llvm::StringTable
// CHECK-NEXT:  MyTargetSDNodeNames = MyTargetSDNodeNamesStorage;
// CHECK-EMPTY:
// CHECK-NEXT:  static const VTByHwModePair MyTargetVTByHwModeTable[] = {
// CHECK-NEXT:   /* dummy */ {0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* dummy */ {SDTCisVT, 0, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/0, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// CHECK-NEXT:      MyTargetVTByHwModeTable, MyTargetSDTypeConstraints);
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace llvm
// CHECK-EMPTY:
// CHECK-NEXT:  #endif // GET_SDNODE_DESC
