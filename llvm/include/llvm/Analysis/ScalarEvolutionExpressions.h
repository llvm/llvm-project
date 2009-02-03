//===- llvm/Analysis/ScalarEvolutionExpressions.h - SCEV Exprs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to represent and build scalar expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_EXPRESSIONS_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_EXPRESSIONS_H

#include "llvm/Analysis/ScalarEvolution.h"

namespace llvm {
  class ConstantInt;
  class ConstantRange;
  class APInt;

  enum SCEVTypes {
    // These should be ordered in terms of increasing complexity to make the
    // folders simpler.
    scConstant, scTruncate, scZeroExtend, scSignExtend, scAddExpr, scMulExpr,
    scUDivExpr, scAddRecExpr, scUMaxExpr, scSMaxExpr, scUnknown,
    scCouldNotCompute
  };

  //===--------------------------------------------------------------------===//
  /// SCEVConstant - This class represents a constant integer value.
  ///
  class SCEVConstant : public SCEV {
    friend class ScalarEvolution;

    ConstantInt *V;
    explicit SCEVConstant(ConstantInt *v) : SCEV(scConstant), V(v) {}

    virtual ~SCEVConstant();
  public:
    ConstantInt *getValue() const { return V; }

    virtual bool isLoopInvariant(const Loop *L) const {
      return true;
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return false;  // Not loop variant
    }

    virtual const Type *getType() const;

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const {
      return this;
    }

    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVConstant *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scConstant;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVTruncateExpr - This class represents a truncation of an integer value
  /// to a smaller integer value.
  ///
  class SCEVTruncateExpr : public SCEV {
    friend class ScalarEvolution;

    SCEVHandle Op;
    const Type *Ty;
    SCEVTruncateExpr(const SCEVHandle &op, const Type *ty);
    virtual ~SCEVTruncateExpr();
  public:
    const SCEVHandle &getOperand() const { return Op; }
    virtual const Type *getType() const { return Ty; }

    virtual bool isLoopInvariant(const Loop *L) const {
      return Op->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return Op->hasComputableLoopEvolution(L);
    }

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const {
      SCEVHandle H = Op->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
      if (H == Op)
        return this;
      return SE.getTruncateExpr(H, Ty);
    }

    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVTruncateExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scTruncate;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVZeroExtendExpr - This class represents a zero extension of a small
  /// integer value to a larger integer value.
  ///
  class SCEVZeroExtendExpr : public SCEV {
    friend class ScalarEvolution;

    SCEVHandle Op;
    const Type *Ty;
    SCEVZeroExtendExpr(const SCEVHandle &op, const Type *ty);
    virtual ~SCEVZeroExtendExpr();
  public:
    const SCEVHandle &getOperand() const { return Op; }
    virtual const Type *getType() const { return Ty; }

    virtual bool isLoopInvariant(const Loop *L) const {
      return Op->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return Op->hasComputableLoopEvolution(L);
    }

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const {
      SCEVHandle H = Op->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
      if (H == Op)
        return this;
      return SE.getZeroExtendExpr(H, Ty);
    }

    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVZeroExtendExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scZeroExtend;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVSignExtendExpr - This class represents a sign extension of a small
  /// integer value to a larger integer value.
  ///
  class SCEVSignExtendExpr : public SCEV {
    friend class ScalarEvolution;

    SCEVHandle Op;
    const Type *Ty;
    SCEVSignExtendExpr(const SCEVHandle &op, const Type *ty);
    virtual ~SCEVSignExtendExpr();
  public:
    const SCEVHandle &getOperand() const { return Op; }
    virtual const Type *getType() const { return Ty; }

    virtual bool isLoopInvariant(const Loop *L) const {
      return Op->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return Op->hasComputableLoopEvolution(L);
    }

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const {
      SCEVHandle H = Op->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
      if (H == Op)
        return this;
      return SE.getSignExtendExpr(H, Ty);
    }

    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVSignExtendExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scSignExtend;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVCommutativeExpr - This node is the base class for n'ary commutative
  /// operators.
  ///
  class SCEVCommutativeExpr : public SCEV {
    friend class ScalarEvolution;

    std::vector<SCEVHandle> Operands;

  protected:
    SCEVCommutativeExpr(enum SCEVTypes T, const std::vector<SCEVHandle> &ops)
      : SCEV(T) {
      Operands.reserve(ops.size());
      Operands.insert(Operands.end(), ops.begin(), ops.end());
    }
    ~SCEVCommutativeExpr();

  public:
    unsigned getNumOperands() const { return (unsigned)Operands.size(); }
    const SCEVHandle &getOperand(unsigned i) const {
      assert(i < Operands.size() && "Operand index out of range!");
      return Operands[i];
    }

    const std::vector<SCEVHandle> &getOperands() const { return Operands; }
    typedef std::vector<SCEVHandle>::const_iterator op_iterator;
    op_iterator op_begin() const { return Operands.begin(); }
    op_iterator op_end() const { return Operands.end(); }


    virtual bool isLoopInvariant(const Loop *L) const {
      for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
        if (!getOperand(i)->isLoopInvariant(L)) return false;
      return true;
    }

    // hasComputableLoopEvolution - Commutative expressions have computable loop
    // evolutions iff they have at least one operand that varies with the loop,
    // but that all varying operands are computable.
    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      bool HasVarying = false;
      for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
        if (!getOperand(i)->isLoopInvariant(L)) {
          if (getOperand(i)->hasComputableLoopEvolution(L))
            HasVarying = true;
          else
            return false;
        }
      return HasVarying;
    }

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const;

    virtual const char *getOperationStr() const = 0;

    virtual const Type *getType() const { return getOperand(0)->getType(); }
    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVCommutativeExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddExpr ||
             S->getSCEVType() == scMulExpr ||
             S->getSCEVType() == scSMaxExpr ||
             S->getSCEVType() == scUMaxExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVAddExpr - This node represents an addition of some number of SCEVs.
  ///
  class SCEVAddExpr : public SCEVCommutativeExpr {
    friend class ScalarEvolution;

    explicit SCEVAddExpr(const std::vector<SCEVHandle> &ops)
      : SCEVCommutativeExpr(scAddExpr, ops) {
    }

  public:
    virtual const char *getOperationStr() const { return " + "; }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVAddExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddExpr;
    }
  };

  //===--------------------------------------------------------------------===//
  /// SCEVMulExpr - This node represents multiplication of some number of SCEVs.
  ///
  class SCEVMulExpr : public SCEVCommutativeExpr {
    friend class ScalarEvolution;

    explicit SCEVMulExpr(const std::vector<SCEVHandle> &ops)
      : SCEVCommutativeExpr(scMulExpr, ops) {
    }

  public:
    virtual const char *getOperationStr() const { return " * "; }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVMulExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scMulExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVUDivExpr - This class represents a binary unsigned division operation.
  ///
  class SCEVUDivExpr : public SCEV {
    friend class ScalarEvolution;

    SCEVHandle LHS, RHS;
    SCEVUDivExpr(const SCEVHandle &lhs, const SCEVHandle &rhs)
      : SCEV(scUDivExpr), LHS(lhs), RHS(rhs) {}

    virtual ~SCEVUDivExpr();
  public:
    const SCEVHandle &getLHS() const { return LHS; }
    const SCEVHandle &getRHS() const { return RHS; }

    virtual bool isLoopInvariant(const Loop *L) const {
      return LHS->isLoopInvariant(L) && RHS->isLoopInvariant(L);
    }

    virtual bool hasComputableLoopEvolution(const Loop *L) const {
      return LHS->hasComputableLoopEvolution(L) &&
             RHS->hasComputableLoopEvolution(L);
    }

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const {
      SCEVHandle L = LHS->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
      SCEVHandle R = RHS->replaceSymbolicValuesWithConcrete(Sym, Conc, SE);
      if (L == LHS && R == RHS)
        return this;
      else
        return SE.getUDivExpr(L, R);
    }


    virtual const Type *getType() const;

    void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUDivExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUDivExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVAddRecExpr - This node represents a polynomial recurrence on the trip
  /// count of the specified loop.
  ///
  /// All operands of an AddRec are required to be loop invariant.
  ///
  class SCEVAddRecExpr : public SCEV {
    friend class ScalarEvolution;

    std::vector<SCEVHandle> Operands;
    const Loop *L;

    SCEVAddRecExpr(const std::vector<SCEVHandle> &ops, const Loop *l)
      : SCEV(scAddRecExpr), Operands(ops), L(l) {
      for (size_t i = 0, e = Operands.size(); i != e; ++i)
        assert(Operands[i]->isLoopInvariant(l) &&
               "Operands of AddRec must be loop-invariant!");
    }
    ~SCEVAddRecExpr();
  public:
    typedef std::vector<SCEVHandle>::const_iterator op_iterator;
    op_iterator op_begin() const { return Operands.begin(); }
    op_iterator op_end() const { return Operands.end(); }

    unsigned getNumOperands() const { return (unsigned)Operands.size(); }
    const SCEVHandle &getOperand(unsigned i) const { return Operands[i]; }
    const SCEVHandle &getStart() const { return Operands[0]; }
    const Loop *getLoop() const { return L; }


    /// getStepRecurrence - This method constructs and returns the recurrence
    /// indicating how much this expression steps by.  If this is a polynomial
    /// of degree N, it returns a chrec of degree N-1.
    SCEVHandle getStepRecurrence(ScalarEvolution &SE) const {
      if (isAffine()) return getOperand(1);
      return SE.getAddRecExpr(std::vector<SCEVHandle>(op_begin()+1,op_end()),
                              getLoop());
    }

    virtual bool hasComputableLoopEvolution(const Loop *QL) const {
      if (L == QL) return true;
      return false;
    }

    virtual bool isLoopInvariant(const Loop *QueryLoop) const;

    virtual const Type *getType() const { return Operands[0]->getType(); }

    /// isAffine - Return true if this is an affine AddRec (i.e., it represents
    /// an expressions A+B*x where A and B are loop invariant values.
    bool isAffine() const {
      // We know that the start value is invariant.  This expression is thus
      // affine iff the step is also invariant.
      return getNumOperands() == 2;
    }

    /// isQuadratic - Return true if this is an quadratic AddRec (i.e., it
    /// represents an expressions A+B*x+C*x^2 where A, B and C are loop
    /// invariant values.  This corresponds to an addrec of the form {L,+,M,+,N}
    bool isQuadratic() const {
      return getNumOperands() == 3;
    }

    /// evaluateAtIteration - Return the value of this chain of recurrences at
    /// the specified iteration number.
    SCEVHandle evaluateAtIteration(SCEVHandle It, ScalarEvolution &SE) const;

    /// getNumIterationsInRange - Return the number of iterations of this loop
    /// that produce values in the specified constant range.  Another way of
    /// looking at this is that it returns the first iteration number where the
    /// value is not in the condition, thus computing the exit count.  If the
    /// iteration count can't be computed, an instance of SCEVCouldNotCompute is
    /// returned.
    SCEVHandle getNumIterationsInRange(ConstantRange Range,
                                       ScalarEvolution &SE) const;

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const;

    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVAddRecExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scAddRecExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVSMaxExpr - This class represents a signed maximum selection.
  ///
  class SCEVSMaxExpr : public SCEVCommutativeExpr {
    friend class ScalarEvolution;

    explicit SCEVSMaxExpr(const std::vector<SCEVHandle> &ops)
      : SCEVCommutativeExpr(scSMaxExpr, ops) {
    }

  public:
    virtual const char *getOperationStr() const { return " smax "; }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVSMaxExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scSMaxExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVUMaxExpr - This class represents an unsigned maximum selection.
  ///
  class SCEVUMaxExpr : public SCEVCommutativeExpr {
    friend class ScalarEvolution;

    explicit SCEVUMaxExpr(const std::vector<SCEVHandle> &ops)
      : SCEVCommutativeExpr(scUMaxExpr, ops) {
    }

  public:
    virtual const char *getOperationStr() const { return " umax "; }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUMaxExpr *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUMaxExpr;
    }
  };


  //===--------------------------------------------------------------------===//
  /// SCEVUnknown - This means that we are dealing with an entirely unknown SCEV
  /// value, and only represent it as it's LLVM Value.  This is the "bottom"
  /// value for the analysis.
  ///
  class SCEVUnknown : public SCEV {
    friend class ScalarEvolution;

    Value *V;
    explicit SCEVUnknown(Value *v) : SCEV(scUnknown), V(v) {}

  protected:
    ~SCEVUnknown();
  public:
    Value *getValue() const { return V; }

    virtual bool isLoopInvariant(const Loop *L) const;
    virtual bool hasComputableLoopEvolution(const Loop *QL) const {
      return false; // not computable
    }

    SCEVHandle replaceSymbolicValuesWithConcrete(const SCEVHandle &Sym,
                                                 const SCEVHandle &Conc,
                                                 ScalarEvolution &SE) const {
      if (&*Sym == this) return Conc;
      return this;
    }

    virtual const Type *getType() const;

    virtual void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const SCEVUnknown *S) { return true; }
    static inline bool classof(const SCEV *S) {
      return S->getSCEVType() == scUnknown;
    }
  };

  /// SCEVVisitor - This class defines a simple visitor class that may be used
  /// for various SCEV analysis purposes.
  template<typename SC, typename RetVal=void>
  struct SCEVVisitor {
    RetVal visit(SCEV *S) {
      switch (S->getSCEVType()) {
      case scConstant:
        return ((SC*)this)->visitConstant((SCEVConstant*)S);
      case scTruncate:
        return ((SC*)this)->visitTruncateExpr((SCEVTruncateExpr*)S);
      case scZeroExtend:
        return ((SC*)this)->visitZeroExtendExpr((SCEVZeroExtendExpr*)S);
      case scSignExtend:
        return ((SC*)this)->visitSignExtendExpr((SCEVSignExtendExpr*)S);
      case scAddExpr:
        return ((SC*)this)->visitAddExpr((SCEVAddExpr*)S);
      case scMulExpr:
        return ((SC*)this)->visitMulExpr((SCEVMulExpr*)S);
      case scUDivExpr:
        return ((SC*)this)->visitUDivExpr((SCEVUDivExpr*)S);
      case scAddRecExpr:
        return ((SC*)this)->visitAddRecExpr((SCEVAddRecExpr*)S);
      case scSMaxExpr:
        return ((SC*)this)->visitSMaxExpr((SCEVSMaxExpr*)S);
      case scUMaxExpr:
        return ((SC*)this)->visitUMaxExpr((SCEVUMaxExpr*)S);
      case scUnknown:
        return ((SC*)this)->visitUnknown((SCEVUnknown*)S);
      case scCouldNotCompute:
        return ((SC*)this)->visitCouldNotCompute((SCEVCouldNotCompute*)S);
      default:
        assert(0 && "Unknown SCEV type!");
        abort();
      }
    }

    RetVal visitCouldNotCompute(SCEVCouldNotCompute *S) {
      assert(0 && "Invalid use of SCEVCouldNotCompute!");
      abort();
      return RetVal();
    }
  };
}

#endif
