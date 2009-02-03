//===- llvm/Analysis/ScalarEvolution.h - Scalar Evolution -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ScalarEvolution class is an LLVM pass which can be used to analyze and
// catagorize scalar expressions in loops.  It specializes in recognizing
// general induction variables, representing them with the abstract and opaque
// SCEV class.  Given this analysis, trip counts of loops and other important
// properties can be obtained.
//
// This analysis is primarily useful for induction variable substitution and
// strength reduction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_H

#include "llvm/Pass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/DataTypes.h"
#include <iosfwd>

namespace llvm {
  class APInt;
  class ConstantInt;
  class Instruction;
  class Type;
  class ConstantRange;
  class SCEVHandle;
  class ScalarEvolution;

  /// SCEV - This class represent an analyzed expression in the program.  These
  /// are reference counted opaque objects that the client is not allowed to
  /// do much with directly.
  ///
  class SCEV {
    const unsigned SCEVType;      // The SCEV baseclass this node corresponds to
    mutable unsigned RefCount;

    friend class SCEVHandle;
    void addRef() const { ++RefCount; }
    void dropRef() const {
      if (--RefCount == 0)
        delete this;
    }

    SCEV(const SCEV &);            // DO NOT IMPLEMENT
    void operator=(const SCEV &);  // DO NOT IMPLEMENT
  protected:
    virtual ~SCEV();
  public:
    explicit SCEV(unsigned SCEVTy) : SCEVType(SCEVTy), RefCount(0) {}

    unsigned getSCEVType() const { return SCEVType; }

    /// isLoopInvariant - Return true if the value of this SCEV is unchanging in
    /// the specified loop.
    virtual bool isLoopInvariant(const Loop *L) const = 0;

    /// hasComputableLoopEvolution - Return true if this SCEV changes value in a
    /// known way in the specified loop.  This property being true implies that
    /// the value is variant in the loop AND that we can emit an expression to
    /// compute the value of the expression at any particular loop iteration.
    virtual bool hasComputableLoopEvolution(const Loop *L) const = 0;

    /// getType - Return the LLVM type of this SCEV expression.
    ///
    virtual const Type *getType() const = 0;

    /// getBitWidth - Get the bit width of the type, if it has one, 0 otherwise.
    /// 
    uint32_t getBitWidth() const;

    /// isZero - Return true if the expression is a constant zero.
    ///
    bool isZero() const;

    /// replaceSymbolicValuesWithConcrete - If this SCEV internally references
    /// the symbolic value "Sym", construct and return a new SCEV that produces
    /// the same value, but which uses the concrete value Conc instead of the
    /// symbolic value.  If this SCEV does not use the symbolic value, it
    /// returns itself.
    virtual SCEVHandle
    replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                      const SCEVHandle &Conc,
                                      ScalarEvolution &SE) const = 0;

    /// print - Print out the internal representation of this scalar to the
    /// specified stream.  This should really only be used for debugging
    /// purposes.
    virtual void print(std::ostream &OS) const = 0;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// dump - This method is used for debugging.
    ///
    void dump() const;
  };

  inline std::ostream &operator<<(std::ostream &OS, const SCEV &S) {
    S.print(OS);
    return OS;
  }

  /// SCEVCouldNotCompute - An object of this class is returned by queries that
  /// could not be answered.  For example, if you ask for the number of
  /// iterations of a linked-list traversal loop, you will get one of these.
  /// None of the standard SCEV operations are valid on this class, it is just a
  /// marker.
  struct SCEVCouldNotCompute : public SCEV {
    SCEVCouldNotCompute();

    // None of these methods are valid for this object.
    virtual bool isLoopInvariant(const Loop *L) const;
    virtual const Type *getType() const;
    virtual bool hasComputableLoopEvolution(const Loop *L) const;
    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }
    virtual SCEVHandle
    replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                      const SCEVHandle &Conc,
                                      ScalarEvolution &SE) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVCouldNotCompute *S) { return true; }
    static bool classof(const SCEV *S);
  };

  /// SCEVHandle - This class is used to maintain the SCEV object's refcounts,
  /// freeing the objects when the last reference is dropped.
  class SCEVHandle {
    SCEV *S;
    SCEVHandle();  // DO NOT IMPLEMENT
  public:
    SCEVHandle(const SCEV *s) : S(const_cast<SCEV*>(s)) {
      assert(S && "Cannot create a handle to a null SCEV!");
      S->addRef();
    }
    SCEVHandle(const SCEVHandle &RHS) : S(RHS.S) {
      S->addRef();
    }
    ~SCEVHandle() { S->dropRef(); }

    operator SCEV*() const { return S; }

    SCEV &operator*() const { return *S; }
    SCEV *operator->() const { return S; }

    bool operator==(SCEV *RHS) const { return S == RHS; }
    bool operator!=(SCEV *RHS) const { return S != RHS; }

    const SCEVHandle &operator=(SCEV *RHS) {
      if (S != RHS) {
        S->dropRef();
        S = RHS;
        S->addRef();
      }
      return *this;
    }

    const SCEVHandle &operator=(const SCEVHandle &RHS) {
      if (S != RHS.S) {
        S->dropRef();
        S = RHS.S;
        S->addRef();
      }
      return *this;
    }
  };

  template<typename From> struct simplify_type;
  template<> struct simplify_type<const SCEVHandle> {
    typedef SCEV* SimpleType;
    static SimpleType getSimplifiedValue(const SCEVHandle &Node) {
      return Node;
    }
  };
  template<> struct simplify_type<SCEVHandle>
    : public simplify_type<const SCEVHandle> {};

  /// ScalarEvolution - This class is the main scalar evolution driver.  Because
  /// client code (intentionally) can't do much with the SCEV objects directly,
  /// they must ask this class for services.
  ///
  class ScalarEvolution : public FunctionPass {
    void *Impl;    // ScalarEvolution uses the pimpl pattern
  public:
    static char ID; // Pass identification, replacement for typeid
    ScalarEvolution() : FunctionPass(&ID), Impl(0) {}

    /// getSCEV - Return a SCEV expression handle for the full generality of the
    /// specified expression.
    SCEVHandle getSCEV(Value *V) const;

    SCEVHandle getConstant(ConstantInt *V);
    SCEVHandle getConstant(const APInt& Val);
    SCEVHandle getTruncateExpr(const SCEVHandle &Op, const Type *Ty);
    SCEVHandle getZeroExtendExpr(const SCEVHandle &Op, const Type *Ty);
    SCEVHandle getSignExtendExpr(const SCEVHandle &Op, const Type *Ty);
    SCEVHandle getAddExpr(std::vector<SCEVHandle> &Ops);
    SCEVHandle getAddExpr(const SCEVHandle &LHS, const SCEVHandle &RHS) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return getAddExpr(Ops);
    }
    SCEVHandle getAddExpr(const SCEVHandle &Op0, const SCEVHandle &Op1,
                          const SCEVHandle &Op2) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(Op0);
      Ops.push_back(Op1);
      Ops.push_back(Op2);
      return getAddExpr(Ops);
    }
    SCEVHandle getMulExpr(std::vector<SCEVHandle> &Ops);
    SCEVHandle getMulExpr(const SCEVHandle &LHS, const SCEVHandle &RHS) {
      std::vector<SCEVHandle> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return getMulExpr(Ops);
    }
    SCEVHandle getUDivExpr(const SCEVHandle &LHS, const SCEVHandle &RHS);
    SCEVHandle getAddRecExpr(const SCEVHandle &Start, const SCEVHandle &Step,
                             const Loop *L);
    SCEVHandle getAddRecExpr(std::vector<SCEVHandle> &Operands,
                             const Loop *L);
    SCEVHandle getAddRecExpr(const std::vector<SCEVHandle> &Operands,
                             const Loop *L) {
      std::vector<SCEVHandle> NewOp(Operands);
      return getAddRecExpr(NewOp, L);
    }
    SCEVHandle getSMaxExpr(const SCEVHandle &LHS, const SCEVHandle &RHS);
    SCEVHandle getSMaxExpr(std::vector<SCEVHandle> Operands);
    SCEVHandle getUMaxExpr(const SCEVHandle &LHS, const SCEVHandle &RHS);
    SCEVHandle getUMaxExpr(std::vector<SCEVHandle> Operands);
    SCEVHandle getUnknown(Value *V);

    /// getNegativeSCEV - Return the SCEV object corresponding to -V.
    ///
    SCEVHandle getNegativeSCEV(const SCEVHandle &V);

    /// getNotSCEV - Return the SCEV object corresponding to ~V.
    ///
    SCEVHandle getNotSCEV(const SCEVHandle &V);

    /// getMinusSCEV - Return LHS-RHS.
    ///
    SCEVHandle getMinusSCEV(const SCEVHandle &LHS,
                            const SCEVHandle &RHS);

    /// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion
    /// of the input value to the specified type.  If the type must be
    /// extended, it is zero extended.
    SCEVHandle getTruncateOrZeroExtend(const SCEVHandle &V, const Type *Ty);

    /// getIntegerSCEV - Given an integer or FP type, create a constant for the
    /// specified signed integer value and return a SCEV for the constant.
    SCEVHandle getIntegerSCEV(int Val, const Type *Ty);

    /// hasSCEV - Return true if the SCEV for this value has already been
    /// computed.
    bool hasSCEV(Value *V) const;

    /// setSCEV - Insert the specified SCEV into the map of current SCEVs for
    /// the specified value.
    void setSCEV(Value *V, const SCEVHandle &H);

    /// getSCEVAtScope - Return a SCEV expression handle for the specified value
    /// at the specified scope in the program.  The L value specifies a loop
    /// nest to evaluate the expression at, where null is the top-level or a
    /// specified loop is immediately inside of the loop.
    ///
    /// This method can be used to compute the exit value for a variable defined
    /// in a loop by querying what the value will hold in the parent loop.
    ///
    /// If this value is not computable at this scope, a SCEVCouldNotCompute
    /// object is returned.
    SCEVHandle getSCEVAtScope(Value *V, const Loop *L) const;

    /// getIterationCount - If the specified loop has a predictable iteration
    /// count, return it, otherwise return a SCEVCouldNotCompute object.
    SCEVHandle getIterationCount(const Loop *L) const;

    /// hasLoopInvariantIterationCount - Return true if the specified loop has
    /// an analyzable loop-invariant iteration count.
    bool hasLoopInvariantIterationCount(const Loop *L) const;

    /// deleteValueFromRecords - This method should be called by the
    /// client before it removes a Value from the program, to make sure
    /// that no dangling references are left around.
    void deleteValueFromRecords(Value *V) const;

    virtual bool runOnFunction(Function &F);
    virtual void releaseMemory();
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void print(std::ostream &OS, const Module* = 0) const;
    void print(std::ostream *OS, const Module* M = 0) const {
      if (OS) print(*OS, M);
    }
  };
}

#endif
