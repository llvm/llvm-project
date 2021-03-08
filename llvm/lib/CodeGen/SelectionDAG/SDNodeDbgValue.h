//===-- llvm/CodeGen/SDNodeDbgValue.h - SelectionDAG dbg_value --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDDbgValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SELECTIONDAG_SDNODEDBGVALUE_H
#define LLVM_LIB_CODEGEN_SELECTIONDAG_SDNODEDBGVALUE_H

#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/DataTypes.h"
#include <utility>

namespace llvm {

class DIVariable;
class DIExpression;
class SDNode;
class Value;
class raw_ostream;

/// Holds the information for a single machine location through SDISel; either
/// an SDNode, a constant, a stack location, or a virtual register.
class SDDbgOperand {
public:
  enum Kind {
    SDNODE = 0,  ///< Value is the result of an expression.
    CONST = 1,   ///< Value is a constant.
    FRAMEIX = 2, ///< Value is contents of a stack location.
    VREG = 3     ///< Value is a virtual register.
  };
  Kind getKind() const { return kind; }

  /// Returns the SDNode* for a register ref
  SDNode *getSDNode() const {
    assert(kind == SDNODE);
    return u.s.Node;
  }

  /// Returns the ResNo for a register ref
  unsigned getResNo() const {
    assert(kind == SDNODE);
    return u.s.ResNo;
  }

  /// Returns the Value* for a constant
  const Value *getConst() const {
    assert(kind == CONST);
    return u.Const;
  }

  /// Returns the FrameIx for a stack object
  unsigned getFrameIx() const {
    assert(kind == FRAMEIX);
    return u.FrameIx;
  }

  /// Returns the Virtual Register for a VReg
  unsigned getVReg() const {
    assert(kind == VREG);
    return u.VReg;
  }

  static SDDbgOperand fromNode(SDNode *Node, unsigned ResNo) {
    return SDDbgOperand(Node, ResNo);
  }
  static SDDbgOperand fromFrameIdx(unsigned FrameIdx) {
    return SDDbgOperand(FrameIdx, FRAMEIX);
  }
  static SDDbgOperand fromVReg(unsigned VReg) {
    return SDDbgOperand(VReg, VREG);
  }
  static SDDbgOperand fromConst(const Value *Const) {
    return SDDbgOperand(Const);
  }

  bool operator!=(const SDDbgOperand &Other) const { return !(*this == Other); }
  bool operator==(const SDDbgOperand &Other) const {
    if (kind != Other.kind)
      return false;
    switch (kind) {
    case SDNODE:
      return getSDNode() == Other.getSDNode() && getResNo() == Other.getResNo();
    case CONST:
      return getConst() == Other.getConst();
    case VREG:
      return getVReg() == Other.getVReg();
    case FRAMEIX:
      return getFrameIx() == Other.getFrameIx();
    }
  }

private:
  Kind kind;
  union {
    struct {
      SDNode *Node;   ///< Valid for expressions.
      unsigned ResNo; ///< Valid for expressions.
    } s;
    const Value *Const; ///< Valid for constants.
    unsigned FrameIx;   ///< Valid for stack objects.
    unsigned VReg;      ///< Valid for registers.
  } u;

  /// Constructor for non-constants.
  SDDbgOperand(SDNode *N, unsigned R) : kind(SDNODE) {
    u.s.Node = N;
    u.s.ResNo = R;
  }
  /// Constructor for constants.
  SDDbgOperand(const Value *C) : kind(CONST) { u.Const = C; }
  /// Constructor for virtual registers and frame indices.
  SDDbgOperand(unsigned VRegOrFrameIdx, Kind Kind) : kind(Kind) {
    assert((Kind == VREG || Kind == FRAMEIX) &&
           "Invalid SDDbgValue constructor");
    if (kind == VREG)
      u.VReg = VRegOrFrameIdx;
    else
      u.FrameIx = VRegOrFrameIdx;
  }
};

/// Holds the information from a dbg_value node through SDISel.
/// We do not use SDValue here to avoid including its header.
class SDDbgValue {
public:
  // FIXME: These SmallVector sizes were chosen without any kind of performance
  // testing.
  using LocOpVector = SmallVector<SDDbgOperand, 2>;
  using SDNodeVector = SmallVector<SDNode *, 2>;

private:
  LocOpVector LocationOps;
  SDNodeVector SDNodes;
  DIVariable *Var;
  DIExpression *Expr;
  DebugLoc DL;
  unsigned Order;
  bool IsIndirect;
  bool IsVariadic;
  bool Invalid = false;
  bool Emitted = false;

public:
  SDDbgValue(DIVariable *Var, DIExpression *Expr, ArrayRef<SDDbgOperand> L,
             ArrayRef<SDNode *> Dependencies, bool IsIndirect, DebugLoc DL,
             unsigned O, bool IsVariadic)
      : LocationOps(L.begin(), L.end()),
        SDNodes(Dependencies.begin(), Dependencies.end()), Var(Var), Expr(Expr),
        DL(DL), Order(O), IsIndirect(IsIndirect), IsVariadic(IsVariadic) {
    assert(IsVariadic || L.size() == 1);
    assert(!(IsVariadic && IsIndirect));
  }

  /// Returns the DIVariable pointer for the variable.
  DIVariable *getVariable() const { return Var; }

  /// Returns the DIExpression pointer for the expression.
  DIExpression *getExpression() const { return Expr; }

  ArrayRef<SDDbgOperand> getLocationOps() const { return LocationOps; }

  LocOpVector copyLocationOps() const { return LocationOps; }

  // Returns the SDNodes which this SDDbgValue depends on.
  ArrayRef<SDNode *> getSDNodes() const { return SDNodes; }

  SDNodeVector copySDNodes() const { return SDNodes; }

  /// Returns whether this is an indirect value.
  bool isIndirect() const { return IsIndirect; }

  bool isVariadic() const { return IsVariadic; }

  /// Returns the DebugLoc.
  DebugLoc getDebugLoc() const { return DL; }

  /// Returns the SDNodeOrder.  This is the order of the preceding node in the
  /// input.
  unsigned getOrder() const { return Order; }

  /// setIsInvalidated / isInvalidated - Setter / getter of the "Invalidated"
  /// property. A SDDbgValue is invalid if the SDNode that produces the value is
  /// deleted.
  void setIsInvalidated() { Invalid = true; }
  bool isInvalidated() const { return Invalid; }

  /// setIsEmitted / isEmitted - Getter/Setter for flag indicating that this
  /// SDDbgValue has been emitted to an MBB.
  void setIsEmitted() { Emitted = true; }
  bool isEmitted() const { return Emitted; }

  /// clearIsEmitted - Reset Emitted flag, for certain special cases where
  /// dbg.addr is emitted twice.
  void clearIsEmitted() { Emitted = false; }

  LLVM_DUMP_METHOD void dump() const;
  LLVM_DUMP_METHOD void print(raw_ostream &OS) const;
};

/// Holds the information from a dbg_label node through SDISel.
/// We do not use SDValue here to avoid including its header.
class SDDbgLabel {
  MDNode *Label;
  DebugLoc DL;
  unsigned Order;

public:
  SDDbgLabel(MDNode *Label, DebugLoc dl, unsigned O)
      : Label(Label), DL(std::move(dl)), Order(O) {}

  /// Returns the MDNode pointer for the label.
  MDNode *getLabel() const { return Label; }

  /// Returns the DebugLoc.
  DebugLoc getDebugLoc() const { return DL; }

  /// Returns the SDNodeOrder.  This is the order of the preceding node in the
  /// input.
  unsigned getOrder() const { return Order; }
};

} // end llvm namespace

#endif
