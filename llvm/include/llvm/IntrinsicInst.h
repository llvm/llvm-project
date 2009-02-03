//===-- llvm/IntrinsicInst.h - Intrinsic Instruction Wrappers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that make it really easy to deal with intrinsic
// functions with the isa/dyncast family of functions.  In particular, this
// allows you to do things like:
//
//     if (MemCpyInst *MCI = dyn_cast<MemCpyInst>(Inst))
//        ... MCI->getDest() ... MCI->getSource() ...
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTRINSICINST_H
#define LLVM_INTRINSICINST_H

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"

namespace llvm {
  /// IntrinsicInst - A useful wrapper class for inspecting calls to intrinsic
  /// functions.  This allows the standard isa/dyncast/cast functionality to
  /// work with calls to intrinsic functions.
  class IntrinsicInst : public CallInst {
    IntrinsicInst();                      // DO NOT IMPLEMENT
    IntrinsicInst(const IntrinsicInst&);  // DO NOT IMPLEMENT
    void operator=(const IntrinsicInst&); // DO NOT IMPLEMENT
  public:
    /// getIntrinsicID - Return the intrinsic ID of this intrinsic.
    ///
    Intrinsic::ID getIntrinsicID() const {
      return (Intrinsic::ID)getCalledFunction()->getIntrinsicID();
    }
    
    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *) { return true; }
    static inline bool classof(const CallInst *I) {
      if (const Function *CF = I->getCalledFunction())
        return CF->getIntrinsicID() != 0;
      return false;
    }
    static inline bool classof(const Value *V) {
      return isa<CallInst>(V) && classof(cast<CallInst>(V));
    }
  };

  /// DbgInfoIntrinsic - This is the common base class for debug info intrinsics
  ///
  struct DbgInfoIntrinsic : public IntrinsicInst {

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const DbgInfoIntrinsic *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      switch (I->getIntrinsicID()) {
      case Intrinsic::dbg_stoppoint:
      case Intrinsic::dbg_func_start:
      case Intrinsic::dbg_region_start:
      case Intrinsic::dbg_region_end:
      case Intrinsic::dbg_declare:
        return true;
      default: return false;
      }
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
    
    static Value *StripCast(Value *C);
  };

  /// DbgStopPointInst - This represents the llvm.dbg.stoppoint instruction.
  ///
  struct DbgStopPointInst : public DbgInfoIntrinsic {
    Value *getLineValue() const { return const_cast<Value*>(getOperand(1)); }
    Value *getColumnValue() const { return const_cast<Value*>(getOperand(2)); }
    Value *getContext() const {
      return StripCast(getOperand(3));
    }

    unsigned getLine() const {
      return unsigned(cast<ConstantInt>(getOperand(1))->getZExtValue());
    }
    unsigned getColumn() const {
      return unsigned(cast<ConstantInt>(getOperand(2))->getZExtValue());
    }
    
    Value* getFileName() const;
    Value* getDirectory() const;

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const DbgStopPointInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_stoppoint;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };
  
  /// DbgFuncStartInst - This represents the llvm.dbg.func.start instruction.
  ///
  struct DbgFuncStartInst : public DbgInfoIntrinsic {
    Value *getSubprogram() const { return StripCast(getOperand(1)); }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const DbgFuncStartInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_func_start;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// DbgRegionStartInst - This represents the llvm.dbg.region.start
  /// instruction.
  struct DbgRegionStartInst : public DbgInfoIntrinsic {
    Value *getContext() const { return StripCast(getOperand(1)); }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const DbgRegionStartInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_region_start;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// DbgRegionEndInst - This represents the llvm.dbg.region.end instruction.
  ///
  struct DbgRegionEndInst : public DbgInfoIntrinsic {
    Value *getContext() const { return StripCast(getOperand(1)); }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const DbgRegionEndInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_region_end;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// DbgDeclareInst - This represents the llvm.dbg.declare instruction.
  ///
  struct DbgDeclareInst : public DbgInfoIntrinsic {
    Value *getAddress()  const { return getOperand(1); }
    Value *getVariable() const { return StripCast(getOperand(2)); }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const DbgDeclareInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_declare;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemIntrinsic - This is the common base class for memset/memcpy/memmove.
  ///
  struct MemIntrinsic : public IntrinsicInst {
    Value *getRawDest() const { return const_cast<Value*>(getOperand(1)); }

    Value *getLength() const { return const_cast<Value*>(getOperand(3)); }
    ConstantInt *getAlignment() const {
      return cast<ConstantInt>(const_cast<Value*>(getOperand(4)));
    }

    /// getDest - This is just like getRawDest, but it strips off any cast
    /// instructions that feed it, giving the original input.  The returned
    /// value is guaranteed to be a pointer.
    Value *getDest() const { return getRawDest()->stripPointerCasts(); }

    /// set* - Set the specified arguments of the instruction.
    ///
    void setDest(Value *Ptr) {
      assert(getRawDest()->getType() == Ptr->getType() &&
             "setDest called with pointer of wrong type!");
      setOperand(1, Ptr);
    }

    void setLength(Value *L) {
      assert(getLength()->getType() == L->getType() &&
             "setLength called with value of wrong type!");
      setOperand(3, L);
    }
    void setAlignment(ConstantInt *A) {
      assert(getAlignment()->getType() == A->getType() &&
             "setAlignment called with value of wrong type!");
      setOperand(4, A);
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const MemIntrinsic *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      switch (I->getIntrinsicID()) {
      case Intrinsic::memcpy:
      case Intrinsic::memmove:
      case Intrinsic::memset:
        return true;
      default: return false;
      }
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };


  /// MemCpyInst - This class wraps the llvm.memcpy intrinsic.
  ///
  struct MemCpyInst : public MemIntrinsic {
    /// get* - Return the arguments to the instruction.
    ///
    Value *getRawSource() const { return const_cast<Value*>(getOperand(2)); }

    /// getSource - This is just like getRawSource, but it strips off any cast
    /// instructions that feed it, giving the original input.  The returned
    /// value is guaranteed to be a pointer.
    Value *getSource() const { return getRawSource()->stripPointerCasts(); }


    void setSource(Value *Ptr) {
      assert(getRawSource()->getType() == Ptr->getType() &&
             "setSource called with pointer of wrong type!");
      setOperand(2, Ptr);
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const MemCpyInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memcpy;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemMoveInst - This class wraps the llvm.memmove intrinsic.
  ///
  struct MemMoveInst : public MemIntrinsic {
    /// get* - Return the arguments to the instruction.
    ///
    Value *getRawSource() const { return const_cast<Value*>(getOperand(2)); }

    /// getSource - This is just like getRawSource, but it strips off any cast
    /// instructions that feed it, giving the original input.  The returned
    /// value is guaranteed to be a pointer.
    Value *getSource() const { return getRawSource()->stripPointerCasts(); }

    void setSource(Value *Ptr) {
      assert(getRawSource()->getType() == Ptr->getType() &&
             "setSource called with pointer of wrong type!");
      setOperand(2, Ptr);
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const MemMoveInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memmove;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemSetInst - This class wraps the llvm.memset intrinsic.
  ///
  struct MemSetInst : public MemIntrinsic {
    /// get* - Return the arguments to the instruction.
    ///
    Value *getValue() const { return const_cast<Value*>(getOperand(2)); }

    void setValue(Value *Val) {
      assert(getValue()->getType() == Val->getType() &&
             "setSource called with pointer of wrong type!");
      setOperand(2, Val);
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const MemSetInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memset;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// EHSelectorInst - This represents the llvm.eh.selector instruction.
  ///
  struct EHSelectorInst : public IntrinsicInst {
    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const EHSelectorInst *) { return true; }
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::eh_selector_i32 ||
             I->getIntrinsicID() == Intrinsic::eh_selector_i64;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };
  
}

#endif
