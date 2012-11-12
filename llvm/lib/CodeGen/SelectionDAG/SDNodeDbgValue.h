//===-- llvm/CodeGen/SDNodeDbgValue.h - SelectionDAG dbg_value --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDDbgValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SDNODEDBGVALUE_H
#define LLVM_CODEGEN_SDNODEDBGVALUE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MDNode;
class SDNode;
class Value;

/// SDDbgValue - Holds the information from a dbg_value node through SDISel.
/// We do not use SDValue here to avoid including its header.

class SDDbgValue {
public:
  enum DbgValueKind {
    SDNODE = 0,             // value is the result of an expression
    CONST = 1,              // value is a constant
    FRAMEIX = 2             // value is contents of a stack location
  };
private:
  enum DbgValueKind kind;
  union {
    struct {
      SDNode *Node;         // valid for expressions
      unsigned ResNo;       // valid for expressions
    } s;
    const Value *Const;     // valid for constants
    unsigned FrameIx;       // valid for stack objects
  } u;
  MDNode *mdPtr;
  uint64_t Offset;
  DebugLoc DL;
  unsigned Order;
  bool Invalid;
public:
  // Constructor for non-constants.
  SDDbgValue(MDNode *mdP, SDNode *N, unsigned R, uint64_t off, DebugLoc dl,
             unsigned O) : mdPtr(mdP), Offset(off), DL(dl), Order(O),
                           Invalid(false) {
    kind = SDNODE;
    u.s.Node = N;
    u.s.ResNo = R;
  }

  // Constructor for constants.
  SDDbgValue(MDNode *mdP, const Value *C, uint64_t off, DebugLoc dl,
             unsigned O) : 
    mdPtr(mdP), Offset(off), DL(dl), Order(O), Invalid(false) {
    kind = CONST;
    u.Const = C;
  }

  // Constructor for frame indices.
  SDDbgValue(MDNode *mdP, unsigned FI, uint64_t off, DebugLoc dl, unsigned O) : 
    mdPtr(mdP), Offset(off), DL(dl), Order(O), Invalid(false) {
    kind = FRAMEIX;
    u.FrameIx = FI;
  }

  // Returns the kind.
  DbgValueKind getKind() { return kind; }

  // Returns the MDNode pointer.
  MDNode *getMDPtr() { return mdPtr; }

  // Returns the SDNode* for a register ref
  SDNode *getSDNode() { assert (kind==SDNODE); return u.s.Node; }

  // Returns the ResNo for a register ref
  unsigned getResNo() { assert (kind==SDNODE); return u.s.ResNo; }

  // Returns the Value* for a constant
  const Value *getConst() { assert (kind==CONST); return u.Const; }

  // Returns the FrameIx for a stack object
  unsigned getFrameIx() { assert (kind==FRAMEIX); return u.FrameIx; }

  // Returns the offset.
  uint64_t getOffset() { return Offset; }

  // Returns the DebugLoc.
  DebugLoc getDebugLoc() { return DL; }

  // Returns the SDNodeOrder.  This is the order of the preceding node in the
  // input.
  unsigned getOrder() { return Order; }

  // setIsInvalidated / isInvalidated - Setter / getter of the "Invalidated"
  // property. A SDDbgValue is invalid if the SDNode that produces the value is
  // deleted.
  void setIsInvalidated() { Invalid = true; }
  bool isInvalidated() { return Invalid; }
};

} // end llvm namespace

#endif
