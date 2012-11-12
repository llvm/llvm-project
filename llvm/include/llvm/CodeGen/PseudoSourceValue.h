//===-- llvm/CodeGen/PseudoSourceValue.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the PseudoSourceValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PSEUDOSOURCEVALUE_H
#define LLVM_CODEGEN_PSEUDOSOURCEVALUE_H

#include "llvm/Value.h"

namespace llvm {
  class MachineFrameInfo;
  class raw_ostream;

  /// PseudoSourceValue - Special value supplied for machine level alias
  /// analysis. It indicates that a memory access references the functions
  /// stack frame (e.g., a spill slot), below the stack frame (e.g., argument
  /// space), or constant pool.
  class PseudoSourceValue : public Value {
  private:
    /// printCustom - Implement printing for PseudoSourceValue. This is called
    /// from Value::print or Value's operator<<.
    ///
    virtual void printCustom(raw_ostream &O) const;

  public:
    explicit PseudoSourceValue(enum ValueTy Subclass = PseudoSourceValueVal);

    /// isConstant - Test whether the memory pointed to by this
    /// PseudoSourceValue has a constant value.
    ///
    virtual bool isConstant(const MachineFrameInfo *) const;

    /// isAliased - Test whether the memory pointed to by this
    /// PseudoSourceValue may also be pointed to by an LLVM IR Value.
    virtual bool isAliased(const MachineFrameInfo *) const;

    /// mayAlias - Return true if the memory pointed to by this
    /// PseudoSourceValue can ever alias a LLVM IR Value.
    virtual bool mayAlias(const MachineFrameInfo *) const;

    /// classof - Methods for support type inquiry through isa, cast, and
    /// dyn_cast:
    ///
    static inline bool classof(const Value *V) {
      return V->getValueID() == PseudoSourceValueVal ||
             V->getValueID() == FixedStackPseudoSourceValueVal;
    }

    /// A pseudo source value referencing a fixed stack frame entry,
    /// e.g., a spill slot.
    static const PseudoSourceValue *getFixedStack(int FI);

    /// A pseudo source value referencing the area below the stack frame of
    /// a function, e.g., the argument space.
    static const PseudoSourceValue *getStack();

    /// A pseudo source value referencing the global offset table
    /// (or something the like).
    static const PseudoSourceValue *getGOT();

    /// A pseudo source value referencing the constant pool. Since constant
    /// pools are constant, this doesn't need to identify a specific constant
    /// pool entry.
    static const PseudoSourceValue *getConstantPool();

    /// A pseudo source value referencing a jump table. Since jump tables are
    /// constant, this doesn't need to identify a specific jump table.
    static const PseudoSourceValue *getJumpTable();
  };

  /// FixedStackPseudoSourceValue - A specialized PseudoSourceValue
  /// for holding FixedStack values, which must include a frame
  /// index.
  class FixedStackPseudoSourceValue : public PseudoSourceValue {
    const int FI;
  public:
    explicit FixedStackPseudoSourceValue(int fi) :
        PseudoSourceValue(FixedStackPseudoSourceValueVal), FI(fi) {}

    /// classof - Methods for support type inquiry through isa, cast, and
    /// dyn_cast:
    ///
    static inline bool classof(const Value *V) {
      return V->getValueID() == FixedStackPseudoSourceValueVal;
    }

    virtual bool isConstant(const MachineFrameInfo *MFI) const;

    virtual bool isAliased(const MachineFrameInfo *MFI) const;

    virtual bool mayAlias(const MachineFrameInfo *) const;

    virtual void printCustom(raw_ostream &OS) const;

    int getFrameIndex() const { return FI; }
  };
} // End llvm namespace

#endif
