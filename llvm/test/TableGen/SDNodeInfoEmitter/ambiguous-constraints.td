// RUN: split-file %s %t

//--- test1.td
// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %t/test1.td | FileCheck %t/test1.td

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

// CHECK:       static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* dummy */ {SDTCisVT, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:      {1, 0, 0, 0, 0, 1, 0, 0}, // NODE
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs,
// CHECK-NEXT:      MyTargetSDNodeNames, MyTargetSDTypeConstraints);


//--- test2.td
// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %t/test2.td | FileCheck %t/test2.td

include "llvm/Target/Target.td"

def MyTarget : Target;

def my_node_1a : SDNode<"MyTargetISD::NODE_1", SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>>;
def my_node_1b : SDNode<"MyTargetISD::NODE_1", SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>>;
def my_node_2a : SDNode<"MyTargetISD::NODE_2", SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>>;
def my_node_2b : SDNode<"MyTargetISD::NODE_2", SDTypeProfile<1, 0, [SDTCisVT<0, untyped>]>>;

// CHECK:       namespace llvm::MyTargetISD {
// CHECK-EMPTY:
// CHECK-NEXT:  enum GenNodeType : unsigned {
// CHECK-NEXT:    NODE_1 = ISD::BUILTIN_OP_END,
// CHECK-NEXT:    NODE_2,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr unsigned GENERATED_OPCODE_END = NODE_2 + 1;
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace llvm::MyTargetISD

// CHECK:       static constexpr char MyTargetSDNodeNamesStorage[] =
// CHECK-NEXT:    "\0"
// CHECK-NEXT:    "MyTargetISD::NODE_1\0"
// CHECK-NEXT:    "MyTargetISD::NODE_2\0"
// CHECK-NEXT:    ;

// CHECK:       static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* 0 */ {SDTCisVT, 0, 0, MVT::i32},
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:      {1, 0, 0, 0, 0, 1, 0, 1}, // NODE_1
// CHECK-NEXT:      {1, 0, 0, 0, 0, 21, 0, 0}, // NODE_2
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/2, MyTargetSDNodeDescs,
// CHECK-NEXT:      MyTargetSDNodeNames, MyTargetSDTypeConstraints);
