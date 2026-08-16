// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s -sdnode-namespace=EmptyISD \
// RUN:   | FileCheck %s -check-prefix=EMPTY

// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s \
// RUN:   | FileCheck %s --check-prefixes=COMMON,TARGET -DNS=MyTargetISD
// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s -sdnode-namespace=MyCustomISD \
// RUN:   | FileCheck %s -check-prefixes=COMMON,CUSTOM -DNS=MyCustomISD

include "llvm/Target/Target.td"

def MyTarget : Target;

def node_1 : SDNode<"MyTargetISD::NODE", SDTypeProfile<1, 0, [SDTCisVT<0, i1>]>>;
def node_2 : SDNode<"MyCustomISD::NODE", SDTypeProfile<0, 1, [SDTCisVT<0, i2>]>>;

// EMPTY:        namespace llvm::EmptyISD {
// EMPTY-EMPTY:
// EMPTY-NEXT:   static constexpr unsigned GENERATED_OPCODE_END = ISD::BUILTIN_OP_END;
// EMPTY-EMPTY:
// EMPTY-NEXT:   } // namespace llvm::EmptyISD

// EMPTY:        static constexpr char MyTargetSDNodeNamesStorage[] =
// EMPTY-NEXT:     "\0"
// EMPTY-NEXT:     ;

// EMPTY:        static const SDNodeDesc MyTargetSDNodeDescs[] = {
// EMPTY-NEXT:   };
// EMPTY-EMPTY:
// EMPTY-NEXT:   #ifndef NDEBUG
// EMPTY:        static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// EMPTY-NEXT:     /* dummy */ {SDTCisVT, 0, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// EMPTY-NEXT:   };
// EMPTY-EMPTY:
// EMPTY-NEXT:   static const SDNodeVerifyDesc MyTargetSDNodeVerifyDescs[] = {
// EMPTY-NEXT:   };
// EMPTY-EMPTY:
// EMPTY-NEXT:   static const SDNodeVerifyInfo MyTargetSDNodeVerifyInfo = {
// EMPTY-NEXT:       MyTargetSDNodeVerifyDescs, MyTargetVTByHwModeTable,
// EMPTY-NEXT:       MyTargetSDTypeConstraints};
// EMPTY-NEXT:   #endif // NDEBUG
// EMPTY-EMPTY:
// EMPTY-NEXT:   #ifndef NDEBUG
// EMPTY-NEXT:   static const SDNodeInfo MyTargetGenSDNodeInfo(
// EMPTY-NEXT:       /*NumOpcodes=*/0, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// EMPTY-NEXT:       &MyTargetSDNodeVerifyInfo);
// EMPTY-NEXT:   #else
// EMPTY-NEXT:   static const SDNodeInfo MyTargetGenSDNodeInfo(
// EMPTY-NEXT:       /*NumOpcodes=*/0, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// EMPTY-NEXT:       nullptr);
// EMPTY-NEXT:   #endif // NDEBUG

// COMMON:       namespace llvm::[[NS]] {
// COMMON-EMPTY:
// COMMON-NEXT:  enum GenNodeType : unsigned {
// COMMON-NEXT:    NODE = ISD::BUILTIN_OP_END,
// COMMON-NEXT:  };
// COMMON-EMPTY:
// COMMON-NEXT:  static constexpr unsigned GENERATED_OPCODE_END = NODE + 1;
// COMMON-EMPTY:
// COMMON-NEXT:  } // namespace llvm::[[NS]]

// COMMON:       static constexpr char MyTargetSDNodeNamesStorage[] =
// COMMON-NEXT:    "\0"
// COMMON-NEXT:    "[[NS]]::NODE\0"
// COMMON-NEXT:    ;

// COMMON:       static const SDNodeDesc MyTargetSDNodeDescs[] = {
// TARGET-NEXT:      {1, 0, 0}, // NODE
// CUSTOM-NEXT:      {1, 0, 0}, // NODE
// COMMON-NEXT:  };
// COMMON-EMPTY:
// COMMON-NEXT:  #ifndef NDEBUG
// COMMON-NEXT:  static const VTByHwModePair MyTargetVTByHwModeTable[] = {
// COMMON-NEXT:    /* dummy */ {0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// COMMON-NEXT:  };

// COMMON:       static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// TARGET-NEXT:    /* 0 */ {SDTCisVT, 0, 0, 0, MVT::i1},
// CUSTOM-NEXT:    /* 0 */ {SDTCisVT, 0, 0, 0, MVT::i2},
// COMMON-NEXT:  };
// COMMON-EMPTY:
// COMMON-NEXT:  static const SDNodeVerifyDesc MyTargetSDNodeVerifyDescs[] = {
// TARGET-NEXT:      {1, 0, 0, 0, 1}, // NODE
// CUSTOM-NEXT:      {0, 1, 0, 0, 1}, // NODE
// COMMON-NEXT:  };
// COMMON-EMPTY:
// COMMON-NEXT:  static const SDNodeVerifyInfo MyTargetSDNodeVerifyInfo = {
// COMMON-NEXT:      MyTargetSDNodeVerifyDescs, MyTargetVTByHwModeTable,
// COMMON-NEXT:      MyTargetSDTypeConstraints};
// COMMON-NEXT:  #endif // NDEBUG
// COMMON-EMPTY:
// COMMON-NEXT:  #ifndef NDEBUG
// COMMON-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// COMMON-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// COMMON-NEXT:      &MyTargetSDNodeVerifyInfo);
// COMMON-NEXT:  #else
// COMMON-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// COMMON-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs, MyTargetSDNodeNames,
// COMMON-NEXT:      nullptr);
// COMMON-NEXT:  #endif // NDEBUG
