// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s 2> %t.warn | FileCheck %s
// RUN: FileCheck --check-prefix=WARN --implicit-check-not=warning %s < %t.warn

// RUN: llvm-tblgen -gen-sd-node-info -warn-on-skipped-nodes=false \
// RUN:   -I %p/../../../include %s 2> %t.nowarn | FileCheck %s
// RUN: not test -s %t.nowarn

include "llvm/Target/Target.td"

def MyTarget : Target;

// WARN: [[#@LINE+1]]:5: warning: skipped node: invalid enum name
def bad_name_1 : SDNode<"", SDTypeProfile<0, 0, []>>;

// WARN: [[#@LINE+1]]:5: warning: skipped node: invalid enum name
def bad_name_2 : SDNode<"NODE", SDTypeProfile<0, 0, []>>;

// WARN: [[#@LINE+1]]:5: warning: skipped node: invalid enum name
def bad_name_3 : SDNode<"MyTargetISD::", SDTypeProfile<0, 0, []>>;

// WARN: [[#@LINE+1]]:5: warning: skipped node: invalid enum name
def bad_name_4 : SDNode<"MyISD::", SDTypeProfile<0, 0, []>>;

// WARN: [[#@LINE+1]]:5: warning: skipped node: invalid enum name
def bad_name_5 : SDNode<"::NODE", SDTypeProfile<0, 0, []>>;


// Standard namespace.
def silent_1 : SDNode<"ISD::SILENT", SDTypeProfile<0, 0, []>>;

// Different namespace.
def silent_2 : SDNode<"MyISD::SILENT", SDTypeProfile<0, 0, []>>;


// Different number of results.
// WARN: [[#@LINE+2]]:5: warning: skipped node: incompatible description
// WARN: [[#@LINE+2]]:5: warning: skipped node: incompatible description
def node_1a : SDNode<"MyTargetISD::NODE_1", SDTypeProfile<0, 0, []>>;
def node_1b : SDNode<"MyTargetISD::NODE_1", SDTypeProfile<1, 0, []>>;

// Different number of operands.
// WARN: [[#@LINE+2]]:5: warning: skipped node: incompatible description
// WARN: [[#@LINE+2]]:5: warning: skipped node: incompatible description
def node_2a : SDNode<"MyTargetISD::NODE_2", SDTypeProfile<0, 0, []>>;
def node_2b : SDNode<"MyTargetISD::NODE_2", SDTypeProfile<0, 1, []>>;

// Different value of IsStrictFP.
// WARN: [[#@LINE+3]]:5: warning: skipped node: incompatible description
// WARN: [[#@LINE+3]]:5: warning: skipped node: incompatible description
let IsStrictFP = true in
def node_3a : SDNode<"MyTargetISD::NODE_3", SDTypeProfile<0, 0, []>>;
def node_3b : SDNode<"MyTargetISD::NODE_3", SDTypeProfile<0, 0, []>>;

// Different value of TSFlags.
// WARN: [[#@LINE+3]]:5: warning: skipped node: incompatible description
// WARN: [[#@LINE+3]]:5: warning: skipped node: incompatible description
let TSFlags = 1 in
def node_4a : SDNode<"MyTargetISD::NODE_4", SDTypeProfile<0, 0, []>>;
def node_4b : SDNode<"MyTargetISD::NODE_4", SDTypeProfile<0, 0, []>>;

// Different properties.
// WARN: [[#@LINE+2]]:5: warning: skipped node: incompatible description
// WARN: [[#@LINE+2]]:5: warning: skipped node: incompatible description
def node_5a : SDNode<"MyTargetISD::NODE_5", SDTypeProfile<0, 0, []>>;
def node_5b : SDNode<"MyTargetISD::NODE_5", SDTypeProfile<0, 0, []>, [SDNPHasChain]>;


// CHECK:       enum GenNodeType : unsigned {
// CHECK-NEXT:    COMPAT = ISD::BUILTIN_OP_END,
// CHECK-NEXT:  };

// CHECK:       static constexpr char MyTargetSDNodeNamesStorage[] =
// CHECK-NEXT:    "\0"
// CHECK-NEXT:    "MyTargetISD::COMPAT\0"
// CHECK-NEXT:    ;

// CHECK:       static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* dummy */ {SDTCisVT, 0, 0, MVT::INVALID_SIMPLE_VALUE_TYPE}
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:      {1, -1, 0, 0, 0, 1, 0, 0}, // COMPAT
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeInfo MyTargetGenSDNodeInfo(
// CHECK-NEXT:      /*NumOpcodes=*/1, MyTargetSDNodeDescs,
// CHECK-NEXT:      MyTargetSDNodeNames, MyTargetSDTypeConstraints);

def compat_a : SDNode<"MyTargetISD::COMPAT", SDTypeProfile<1, -1, []>>;
def compat_b : SDNode<"MyTargetISD::COMPAT", SDTypeProfile<1, -1, [SDTCisVT<0, untyped>]>>;
def compat_c : SDNode<"MyTargetISD::COMPAT", SDTypeProfile<1, -1, [SDTCisVT<0, untyped>]>,
    [SDNPCommutative, SDNPAssociative, SDNPMayStore, SDNPMayLoad, SDNPSideEffect]>;
