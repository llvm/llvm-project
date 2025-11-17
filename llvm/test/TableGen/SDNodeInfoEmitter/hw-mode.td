// RUN: llvm-tblgen -gen-sd-node-info -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

def M1 : HwMode<[]>;
def M2 : HwMode<[]>;
def M3 : HwMode<[]>;

def VT1 : ValueTypeByHwMode<[M1], [i1]>;
def VT2 : ValueTypeByHwMode<[M2], [i2]>;
def VT3 : ValueTypeByHwMode<[M1, M2, M3, DefaultMode], [i1, i2, i4, i8]>;

def my_node_1 : SDNode<
    "MyTargetISD::NODE_1",
    SDTypeProfile<0, 5, [
        SDTCVecEltisVT<0, VT3>,
        SDTCisVT<1, i1>,
        SDTCVecEltisVT<2, i2>,
        SDTCisVT<3, VT1>,
        SDTCVecEltisVT<4, VT2>,
    ]>
>;

def my_node_2 : SDNode<
    "MyTargetISD::NODE_2",
    SDTypeProfile<1, 2, [
        SDTCVecEltisVT<0, VT3>,
        SDTCisVT<1, i1>,
        SDTCVecEltisVT<2, i2>,
    ]>
>;

def my_node_3 : SDNode<
    "MyTargetISD::NODE_3",
    SDTypeProfile<1, 0, [
        SDTCisVT<0, VT3>,
    ]>
>;

// CHECK:       static const VTByHwModePair MyTargetVTByHwModeTable[] = {
// CHECK-NEXT:    /* 0 */ {0, MVT::i8}, {1, MVT::i1}, {2, MVT::i2}, {3, MVT::i4},
// CHECK-NEXT:    /* 4 */ {1, MVT::i1},
// CHECK-NEXT:    /* 5 */ {2, MVT::i2},
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDTypeConstraint MyTargetSDTypeConstraints[] = {
// CHECK-NEXT:    /* 0 */ {SDTCisVT, 0, 0, 4, 0},
// CHECK-NEXT:    /* 1 */ {SDTCVecEltisVT, 4, 0, 1, 5},
// CHECK-SAME:            {SDTCisVT, 3, 0, 1, 4},
// CHECK-SAME:            {SDTCVecEltisVT, 2, 0, 0, MVT::i2},
// CHECK-SAME:            {SDTCisVT, 1, 0, 0, MVT::i1},
// CHECK-SAME:            {SDTCVecEltisVT, 0, 0, 4, 0},
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static const SDNodeDesc MyTargetSDNodeDescs[] = {
// CHECK-NEXT:      {0, 5, 0, 0, 0, 1, 1, 5}, // NODE_1
// CHECK-NEXT:      {1, 2, 0, 0, 0, 21, 3, 3}, // NODE_2
// CHECK-NEXT:      {1, 0, 0, 0, 0, 41, 0, 1}, // NODE_3
// CHECK-NEXT:  };
