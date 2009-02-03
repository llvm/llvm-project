//===- GVN.cpp - Eliminate redundant values and loads ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global value numbering to eliminate fully redundant
// instructions.  It also performs simple dead load elimination.
//
// Note that this pass does the value numbering itself, it does not use the
// ValueNumbering analysis passes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gvn"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cstdio>
using namespace llvm;

STATISTIC(NumGVNInstr,  "Number of instructions deleted");
STATISTIC(NumGVNLoad,   "Number of loads deleted");
STATISTIC(NumGVNPRE,    "Number of instructions PRE'd");
STATISTIC(NumGVNBlocks, "Number of blocks merged");
STATISTIC(NumPRELoad,   "Number of loads PRE'd");

static cl::opt<bool> EnablePRE("enable-pre",
                               cl::init(true), cl::Hidden);
cl::opt<bool> EnableLoadPRE("enable-load-pre"/*, cl::init(true)*/);

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.
namespace {
  struct VISIBILITY_HIDDEN Expression {
    enum ExpressionOpcode { ADD, SUB, MUL, UDIV, SDIV, FDIV, UREM, SREM, 
                            FREM, SHL, LSHR, ASHR, AND, OR, XOR, ICMPEQ, 
                            ICMPNE, ICMPUGT, ICMPUGE, ICMPULT, ICMPULE, 
                            ICMPSGT, ICMPSGE, ICMPSLT, ICMPSLE, FCMPOEQ, 
                            FCMPOGT, FCMPOGE, FCMPOLT, FCMPOLE, FCMPONE, 
                            FCMPORD, FCMPUNO, FCMPUEQ, FCMPUGT, FCMPUGE, 
                            FCMPULT, FCMPULE, FCMPUNE, EXTRACT, INSERT,
                            SHUFFLE, SELECT, TRUNC, ZEXT, SEXT, FPTOUI,
                            FPTOSI, UITOFP, SITOFP, FPTRUNC, FPEXT, 
                            PTRTOINT, INTTOPTR, BITCAST, GEP, CALL, CONSTANT,
                            EMPTY, TOMBSTONE };

    ExpressionOpcode opcode;
    const Type* type;
    uint32_t firstVN;
    uint32_t secondVN;
    uint32_t thirdVN;
    SmallVector<uint32_t, 4> varargs;
    Value* function;
  
    Expression() { }
    Expression(ExpressionOpcode o) : opcode(o) { }
  
    bool operator==(const Expression &other) const {
      if (opcode != other.opcode)
        return false;
      else if (opcode == EMPTY || opcode == TOMBSTONE)
        return true;
      else if (type != other.type)
        return false;
      else if (function != other.function)
        return false;
      else if (firstVN != other.firstVN)
        return false;
      else if (secondVN != other.secondVN)
        return false;
      else if (thirdVN != other.thirdVN)
        return false;
      else {
        if (varargs.size() != other.varargs.size())
          return false;
      
        for (size_t i = 0; i < varargs.size(); ++i)
          if (varargs[i] != other.varargs[i])
            return false;
    
        return true;
      }
    }
  
    bool operator!=(const Expression &other) const {
      return !(*this == other);
    }
  };
  
  class VISIBILITY_HIDDEN ValueTable {
    private:
      DenseMap<Value*, uint32_t> valueNumbering;
      DenseMap<Expression, uint32_t> expressionNumbering;
      AliasAnalysis* AA;
      MemoryDependenceAnalysis* MD;
      DominatorTree* DT;
  
      uint32_t nextValueNumber;
    
      Expression::ExpressionOpcode getOpcode(BinaryOperator* BO);
      Expression::ExpressionOpcode getOpcode(CmpInst* C);
      Expression::ExpressionOpcode getOpcode(CastInst* C);
      Expression create_expression(BinaryOperator* BO);
      Expression create_expression(CmpInst* C);
      Expression create_expression(ShuffleVectorInst* V);
      Expression create_expression(ExtractElementInst* C);
      Expression create_expression(InsertElementInst* V);
      Expression create_expression(SelectInst* V);
      Expression create_expression(CastInst* C);
      Expression create_expression(GetElementPtrInst* G);
      Expression create_expression(CallInst* C);
      Expression create_expression(Constant* C);
    public:
      ValueTable() : nextValueNumber(1) { }
      uint32_t lookup_or_add(Value* V);
      uint32_t lookup(Value* V) const;
      void add(Value* V, uint32_t num);
      void clear();
      void erase(Value* v);
      unsigned size();
      void setAliasAnalysis(AliasAnalysis* A) { AA = A; }
      AliasAnalysis *getAliasAnalysis() const { return AA; }
      void setMemDep(MemoryDependenceAnalysis* M) { MD = M; }
      void setDomTree(DominatorTree* D) { DT = D; }
      uint32_t getNextUnusedValueNumber() { return nextValueNumber; }
      void verifyRemoved(const Value *) const;
  };
}

namespace llvm {
template <> struct DenseMapInfo<Expression> {
  static inline Expression getEmptyKey() {
    return Expression(Expression::EMPTY);
  }
  
  static inline Expression getTombstoneKey() {
    return Expression(Expression::TOMBSTONE);
  }
  
  static unsigned getHashValue(const Expression e) {
    unsigned hash = e.opcode;
    
    hash = e.firstVN + hash * 37;
    hash = e.secondVN + hash * 37;
    hash = e.thirdVN + hash * 37;
    
    hash = ((unsigned)((uintptr_t)e.type >> 4) ^
            (unsigned)((uintptr_t)e.type >> 9)) +
           hash * 37;
    
    for (SmallVector<uint32_t, 4>::const_iterator I = e.varargs.begin(),
         E = e.varargs.end(); I != E; ++I)
      hash = *I + hash * 37;
    
    hash = ((unsigned)((uintptr_t)e.function >> 4) ^
            (unsigned)((uintptr_t)e.function >> 9)) +
           hash * 37;
    
    return hash;
  }
  static bool isEqual(const Expression &LHS, const Expression &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return true; }
};
}

//===----------------------------------------------------------------------===//
//                     ValueTable Internal Functions
//===----------------------------------------------------------------------===//
Expression::ExpressionOpcode ValueTable::getOpcode(BinaryOperator* BO) {
  switch(BO->getOpcode()) {
  default: // THIS SHOULD NEVER HAPPEN
    assert(0 && "Binary operator with unknown opcode?");
  case Instruction::Add:  return Expression::ADD;
  case Instruction::Sub:  return Expression::SUB;
  case Instruction::Mul:  return Expression::MUL;
  case Instruction::UDiv: return Expression::UDIV;
  case Instruction::SDiv: return Expression::SDIV;
  case Instruction::FDiv: return Expression::FDIV;
  case Instruction::URem: return Expression::UREM;
  case Instruction::SRem: return Expression::SREM;
  case Instruction::FRem: return Expression::FREM;
  case Instruction::Shl:  return Expression::SHL;
  case Instruction::LShr: return Expression::LSHR;
  case Instruction::AShr: return Expression::ASHR;
  case Instruction::And:  return Expression::AND;
  case Instruction::Or:   return Expression::OR;
  case Instruction::Xor:  return Expression::XOR;
  }
}

Expression::ExpressionOpcode ValueTable::getOpcode(CmpInst* C) {
  if (isa<ICmpInst>(C) || isa<VICmpInst>(C)) {
    switch (C->getPredicate()) {
    default:  // THIS SHOULD NEVER HAPPEN
      assert(0 && "Comparison with unknown predicate?");
    case ICmpInst::ICMP_EQ:  return Expression::ICMPEQ;
    case ICmpInst::ICMP_NE:  return Expression::ICMPNE;
    case ICmpInst::ICMP_UGT: return Expression::ICMPUGT;
    case ICmpInst::ICMP_UGE: return Expression::ICMPUGE;
    case ICmpInst::ICMP_ULT: return Expression::ICMPULT;
    case ICmpInst::ICMP_ULE: return Expression::ICMPULE;
    case ICmpInst::ICMP_SGT: return Expression::ICMPSGT;
    case ICmpInst::ICMP_SGE: return Expression::ICMPSGE;
    case ICmpInst::ICMP_SLT: return Expression::ICMPSLT;
    case ICmpInst::ICMP_SLE: return Expression::ICMPSLE;
    }
  }
  assert((isa<FCmpInst>(C) || isa<VFCmpInst>(C)) && "Unknown compare");
  switch (C->getPredicate()) {
  default: // THIS SHOULD NEVER HAPPEN
    assert(0 && "Comparison with unknown predicate?");
  case FCmpInst::FCMP_OEQ: return Expression::FCMPOEQ;
  case FCmpInst::FCMP_OGT: return Expression::FCMPOGT;
  case FCmpInst::FCMP_OGE: return Expression::FCMPOGE;
  case FCmpInst::FCMP_OLT: return Expression::FCMPOLT;
  case FCmpInst::FCMP_OLE: return Expression::FCMPOLE;
  case FCmpInst::FCMP_ONE: return Expression::FCMPONE;
  case FCmpInst::FCMP_ORD: return Expression::FCMPORD;
  case FCmpInst::FCMP_UNO: return Expression::FCMPUNO;
  case FCmpInst::FCMP_UEQ: return Expression::FCMPUEQ;
  case FCmpInst::FCMP_UGT: return Expression::FCMPUGT;
  case FCmpInst::FCMP_UGE: return Expression::FCMPUGE;
  case FCmpInst::FCMP_ULT: return Expression::FCMPULT;
  case FCmpInst::FCMP_ULE: return Expression::FCMPULE;
  case FCmpInst::FCMP_UNE: return Expression::FCMPUNE;
  }
}

Expression::ExpressionOpcode ValueTable::getOpcode(CastInst* C) {
  switch(C->getOpcode()) {
  default: // THIS SHOULD NEVER HAPPEN
    assert(0 && "Cast operator with unknown opcode?");
  case Instruction::Trunc:    return Expression::TRUNC;
  case Instruction::ZExt:     return Expression::ZEXT;
  case Instruction::SExt:     return Expression::SEXT;
  case Instruction::FPToUI:   return Expression::FPTOUI;
  case Instruction::FPToSI:   return Expression::FPTOSI;
  case Instruction::UIToFP:   return Expression::UITOFP;
  case Instruction::SIToFP:   return Expression::SITOFP;
  case Instruction::FPTrunc:  return Expression::FPTRUNC;
  case Instruction::FPExt:    return Expression::FPEXT;
  case Instruction::PtrToInt: return Expression::PTRTOINT;
  case Instruction::IntToPtr: return Expression::INTTOPTR;
  case Instruction::BitCast:  return Expression::BITCAST;
  }
}

Expression ValueTable::create_expression(CallInst* C) {
  Expression e;
  
  e.type = C->getType();
  e.firstVN = 0;
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = C->getCalledFunction();
  e.opcode = Expression::CALL;
  
  for (CallInst::op_iterator I = C->op_begin()+1, E = C->op_end();
       I != E; ++I)
    e.varargs.push_back(lookup_or_add(*I));
  
  return e;
}

Expression ValueTable::create_expression(BinaryOperator* BO) {
  Expression e;
    
  e.firstVN = lookup_or_add(BO->getOperand(0));
  e.secondVN = lookup_or_add(BO->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = BO->getType();
  e.opcode = getOpcode(BO);
  
  return e;
}

Expression ValueTable::create_expression(CmpInst* C) {
  Expression e;
    
  e.firstVN = lookup_or_add(C->getOperand(0));
  e.secondVN = lookup_or_add(C->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = C->getType();
  e.opcode = getOpcode(C);
  
  return e;
}

Expression ValueTable::create_expression(CastInst* C) {
  Expression e;
    
  e.firstVN = lookup_or_add(C->getOperand(0));
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = 0;
  e.type = C->getType();
  e.opcode = getOpcode(C);
  
  return e;
}

Expression ValueTable::create_expression(ShuffleVectorInst* S) {
  Expression e;
    
  e.firstVN = lookup_or_add(S->getOperand(0));
  e.secondVN = lookup_or_add(S->getOperand(1));
  e.thirdVN = lookup_or_add(S->getOperand(2));
  e.function = 0;
  e.type = S->getType();
  e.opcode = Expression::SHUFFLE;
  
  return e;
}

Expression ValueTable::create_expression(ExtractElementInst* E) {
  Expression e;
    
  e.firstVN = lookup_or_add(E->getOperand(0));
  e.secondVN = lookup_or_add(E->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = E->getType();
  e.opcode = Expression::EXTRACT;
  
  return e;
}

Expression ValueTable::create_expression(InsertElementInst* I) {
  Expression e;
    
  e.firstVN = lookup_or_add(I->getOperand(0));
  e.secondVN = lookup_or_add(I->getOperand(1));
  e.thirdVN = lookup_or_add(I->getOperand(2));
  e.function = 0;
  e.type = I->getType();
  e.opcode = Expression::INSERT;
  
  return e;
}

Expression ValueTable::create_expression(SelectInst* I) {
  Expression e;
    
  e.firstVN = lookup_or_add(I->getCondition());
  e.secondVN = lookup_or_add(I->getTrueValue());
  e.thirdVN = lookup_or_add(I->getFalseValue());
  e.function = 0;
  e.type = I->getType();
  e.opcode = Expression::SELECT;
  
  return e;
}

Expression ValueTable::create_expression(GetElementPtrInst* G) {
  Expression e;
  
  e.firstVN = lookup_or_add(G->getPointerOperand());
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = 0;
  e.type = G->getType();
  e.opcode = Expression::GEP;
  
  for (GetElementPtrInst::op_iterator I = G->idx_begin(), E = G->idx_end();
       I != E; ++I)
    e.varargs.push_back(lookup_or_add(*I));
  
  return e;
}

//===----------------------------------------------------------------------===//
//                     ValueTable External Functions
//===----------------------------------------------------------------------===//

/// add - Insert a value into the table with a specified value number.
void ValueTable::add(Value* V, uint32_t num) {
  valueNumbering.insert(std::make_pair(V, num));
}

/// lookup_or_add - Returns the value number for the specified value, assigning
/// it a new number if it did not have one before.
uint32_t ValueTable::lookup_or_add(Value* V) {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  
  if (CallInst* C = dyn_cast<CallInst>(V)) {
    if (AA->doesNotAccessMemory(C)) {
      Expression e = create_expression(C);
    
      DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
      if (EI != expressionNumbering.end()) {
        valueNumbering.insert(std::make_pair(V, EI->second));
        return EI->second;
      } else {
        expressionNumbering.insert(std::make_pair(e, nextValueNumber));
        valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
        return nextValueNumber++;
      }
    } else if (AA->onlyReadsMemory(C)) {
      Expression e = create_expression(C);
      
      if (expressionNumbering.find(e) == expressionNumbering.end()) {
        expressionNumbering.insert(std::make_pair(e, nextValueNumber));
        valueNumbering.insert(std::make_pair(V, nextValueNumber));
        return nextValueNumber++;
      }
      
      MemDepResult local_dep = MD->getDependency(C);
      
      if (!local_dep.isDef() && !local_dep.isNonLocal()) {
        valueNumbering.insert(std::make_pair(V, nextValueNumber));
        return nextValueNumber++;
      }

      if (local_dep.isDef()) {
        CallInst* local_cdep = cast<CallInst>(local_dep.getInst());
        
        if (local_cdep->getNumOperands() != C->getNumOperands()) {
          valueNumbering.insert(std::make_pair(V, nextValueNumber));
          return nextValueNumber++;
        }
          
        for (unsigned i = 1; i < C->getNumOperands(); ++i) {
          uint32_t c_vn = lookup_or_add(C->getOperand(i));
          uint32_t cd_vn = lookup_or_add(local_cdep->getOperand(i));
          if (c_vn != cd_vn) {
            valueNumbering.insert(std::make_pair(V, nextValueNumber));
            return nextValueNumber++;
          }
        }
      
        uint32_t v = lookup_or_add(local_cdep);
        valueNumbering.insert(std::make_pair(V, v));
        return v;
      }

      // Non-local case.
      const MemoryDependenceAnalysis::NonLocalDepInfo &deps = 
        MD->getNonLocalCallDependency(CallSite(C));
      // FIXME: call/call dependencies for readonly calls should return def, not
      // clobber!  Move the checking logic to MemDep!
      CallInst* cdep = 0;
      
      // Check to see if we have a single dominating call instruction that is
      // identical to C.
      for (unsigned i = 0, e = deps.size(); i != e; ++i) {
        const MemoryDependenceAnalysis::NonLocalDepEntry *I = &deps[i];
        // Ignore non-local dependencies.
        if (I->second.isNonLocal())
          continue;

        // We don't handle non-depedencies.  If we already have a call, reject
        // instruction dependencies.
        if (I->second.isClobber() || cdep != 0) {
          cdep = 0;
          break;
        }
        
        CallInst *NonLocalDepCall = dyn_cast<CallInst>(I->second.getInst());
        // FIXME: All duplicated with non-local case.
        if (NonLocalDepCall && DT->properlyDominates(I->first, C->getParent())){
          cdep = NonLocalDepCall;
          continue;
        }
        
        cdep = 0;
        break;
      }
      
      if (!cdep) {
        valueNumbering.insert(std::make_pair(V, nextValueNumber));
        return nextValueNumber++;
      }
      
      if (cdep->getNumOperands() != C->getNumOperands()) {
        valueNumbering.insert(std::make_pair(V, nextValueNumber));
        return nextValueNumber++;
      }
      for (unsigned i = 1; i < C->getNumOperands(); ++i) {
        uint32_t c_vn = lookup_or_add(C->getOperand(i));
        uint32_t cd_vn = lookup_or_add(cdep->getOperand(i));
        if (c_vn != cd_vn) {
          valueNumbering.insert(std::make_pair(V, nextValueNumber));
          return nextValueNumber++;
        }
      }
      
      uint32_t v = lookup_or_add(cdep);
      valueNumbering.insert(std::make_pair(V, v));
      return v;
      
    } else {
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      return nextValueNumber++;
    }
  } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Expression e = create_expression(BO);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Expression e = create_expression(C);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (ShuffleVectorInst* U = dyn_cast<ShuffleVectorInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (ExtractElementInst* U = dyn_cast<ExtractElementInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (InsertElementInst* U = dyn_cast<InsertElementInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (SelectInst* U = dyn_cast<SelectInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CastInst* U = dyn_cast<CastInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (GetElementPtrInst* U = dyn_cast<GetElementPtrInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else {
    valueNumbering.insert(std::make_pair(V, nextValueNumber));
    return nextValueNumber++;
  }
}

/// lookup - Returns the value number of the specified value. Fails if
/// the value has not yet been numbered.
uint32_t ValueTable::lookup(Value* V) const {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  assert(VI != valueNumbering.end() && "Value not numbered?");
  return VI->second;
}

/// clear - Remove all entries from the ValueTable
void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  nextValueNumber = 1;
}

/// erase - Remove a value from the value numbering
void ValueTable::erase(Value* V) {
  valueNumbering.erase(V);
}

/// verifyRemoved - Verify that the value is removed from all internal data
/// structures.
void ValueTable::verifyRemoved(const Value *V) const {
  for (DenseMap<Value*, uint32_t>::iterator
         I = valueNumbering.begin(), E = valueNumbering.end(); I != E; ++I) {
    assert(I->first != V && "Inst still occurs in value numbering map!");
  }
}

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

namespace {
  struct VISIBILITY_HIDDEN ValueNumberScope {
    ValueNumberScope* parent;
    DenseMap<uint32_t, Value*> table;
    
    ValueNumberScope(ValueNumberScope* p) : parent(p) { }
  };
}

namespace {

  class VISIBILITY_HIDDEN GVN : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVN() : FunctionPass(&ID) { }

  private:
    MemoryDependenceAnalysis *MD;
    DominatorTree *DT;

    ValueTable VN;
    DenseMap<BasicBlock*, ValueNumberScope*> localAvail;
    
    typedef DenseMap<Value*, SmallPtrSet<Instruction*, 4> > PhiMapType;
    PhiMapType phiMap;
    
    
    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addRequired<AliasAnalysis>();
      
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<AliasAnalysis>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    bool processLoad(LoadInst* L,
                     SmallVectorImpl<Instruction*> &toErase);
    bool processInstruction(Instruction* I,
                            SmallVectorImpl<Instruction*> &toErase);
    bool processNonLocalLoad(LoadInst* L,
                             SmallVectorImpl<Instruction*> &toErase);
    bool processBlock(BasicBlock* BB);
    Value *GetValueForBlock(BasicBlock *BB, Instruction* orig,
                            DenseMap<BasicBlock*, Value*> &Phis,
                            bool top_level = false);
    void dump(DenseMap<uint32_t, Value*>& d);
    bool iterateOnFunction(Function &F);
    Value* CollapsePhi(PHINode* p);
    bool isSafeReplacement(PHINode* p, Instruction* inst);
    bool performPRE(Function& F);
    Value* lookupNumber(BasicBlock* BB, uint32_t num);
    bool mergeBlockIntoPredecessor(BasicBlock* BB);
    Value* AttemptRedundancyElimination(Instruction* orig, unsigned valno);
    void cleanupGlobalSets();
    void verifyRemoved(const Instruction *I) const;
  };
  
  char GVN::ID = 0;
}

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createGVNPass() { return new GVN(); }

static RegisterPass<GVN> X("gvn",
                           "Global Value Numbering");

void GVN::dump(DenseMap<uint32_t, Value*>& d) {
  printf("{\n");
  for (DenseMap<uint32_t, Value*>::iterator I = d.begin(),
       E = d.end(); I != E; ++I) {
      printf("%d\n", I->first);
      I->second->dump();
  }
  printf("}\n");
}

Value* GVN::CollapsePhi(PHINode* p) {
  Value* constVal = p->hasConstantValue();
  if (!constVal) return 0;
  
  Instruction* inst = dyn_cast<Instruction>(constVal);
  if (!inst)
    return constVal;
    
  if (DT->dominates(inst, p))
    if (isSafeReplacement(p, inst))
      return inst;
  return 0;
}

bool GVN::isSafeReplacement(PHINode* p, Instruction* inst) {
  if (!isa<PHINode>(inst))
    return true;
  
  for (Instruction::use_iterator UI = p->use_begin(), E = p->use_end();
       UI != E; ++UI)
    if (PHINode* use_phi = dyn_cast<PHINode>(UI))
      if (use_phi->getParent() == inst->getParent())
        return false;
  
  return true;
}

/// GetValueForBlock - Get the value to use within the specified basic block.
/// available values are in Phis.
Value *GVN::GetValueForBlock(BasicBlock *BB, Instruction* orig,
                             DenseMap<BasicBlock*, Value*> &Phis,
                             bool top_level) { 
                                 
  // If we have already computed this value, return the previously computed val.
  DenseMap<BasicBlock*, Value*>::iterator V = Phis.find(BB);
  if (V != Phis.end() && !top_level) return V->second;
  
  // If the block is unreachable, just return undef, since this path
  // can't actually occur at runtime.
  if (!DT->isReachableFromEntry(BB))
    return Phis[BB] = UndefValue::get(orig->getType());
  
  if (BasicBlock *Pred = BB->getSinglePredecessor()) {
    Value *ret = GetValueForBlock(Pred, orig, Phis);
    Phis[BB] = ret;
    return ret;
  }

  // Get the number of predecessors of this block so we can reserve space later.
  // If there is already a PHI in it, use the #preds from it, otherwise count.
  // Getting it from the PHI is constant time.
  unsigned NumPreds;
  if (PHINode *ExistingPN = dyn_cast<PHINode>(BB->begin()))
    NumPreds = ExistingPN->getNumIncomingValues();
  else
    NumPreds = std::distance(pred_begin(BB), pred_end(BB));
  
  // Otherwise, the idom is the loop, so we need to insert a PHI node.  Do so
  // now, then get values to fill in the incoming values for the PHI.
  PHINode *PN = PHINode::Create(orig->getType(), orig->getName()+".rle",
                                BB->begin());
  PN->reserveOperandSpace(NumPreds);
  
  Phis.insert(std::make_pair(BB, PN));
  
  // Fill in the incoming values for the block.
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    Value* val = GetValueForBlock(*PI, orig, Phis);
    PN->addIncoming(val, *PI);
  }
  
  VN.getAliasAnalysis()->copyValue(orig, PN);
  
  // Attempt to collapse PHI nodes that are trivially redundant
  Value* v = CollapsePhi(PN);
  if (!v) {
    // Cache our phi construction results
    if (LoadInst* L = dyn_cast<LoadInst>(orig))
      phiMap[L->getPointerOperand()].insert(PN);
    else
      phiMap[orig].insert(PN);
    
    return PN;
  }
    
  PN->replaceAllUsesWith(v);
  if (isa<PointerType>(v->getType()))
    MD->invalidateCachedPointerInfo(v);

  for (DenseMap<BasicBlock*, Value*>::iterator I = Phis.begin(),
       E = Phis.end(); I != E; ++I)
    if (I->second == PN)
      I->second = v;

  DEBUG(cerr << "GVN removed: " << *PN);
  MD->removeInstruction(PN);
  PN->eraseFromParent();
  DEBUG(verifyRemoved(PN));

  Phis[BB] = v;
  return v;
}

/// IsValueFullyAvailableInBlock - Return true if we can prove that the value
/// we're analyzing is fully available in the specified block.  As we go, keep
/// track of which blocks we know are fully alive in FullyAvailableBlocks.  This
/// map is actually a tri-state map with the following values:
///   0) we know the block *is not* fully available.
///   1) we know the block *is* fully available.
///   2) we do not know whether the block is fully available or not, but we are
///      currently speculating that it will be.
///   3) we are speculating for this block and have used that to speculate for
///      other blocks.
static bool IsValueFullyAvailableInBlock(BasicBlock *BB, 
                            DenseMap<BasicBlock*, char> &FullyAvailableBlocks) {
  // Optimistically assume that the block is fully available and check to see
  // if we already know about this block in one lookup.
  std::pair<DenseMap<BasicBlock*, char>::iterator, char> IV = 
    FullyAvailableBlocks.insert(std::make_pair(BB, 2));

  // If the entry already existed for this block, return the precomputed value.
  if (!IV.second) {
    // If this is a speculative "available" value, mark it as being used for
    // speculation of other blocks.
    if (IV.first->second == 2)
      IV.first->second = 3;
    return IV.first->second != 0;
  }
  
  // Otherwise, see if it is fully available in all predecessors.
  pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
  
  // If this block has no predecessors, it isn't live-in here.
  if (PI == PE)
    goto SpeculationFailure;
  
  for (; PI != PE; ++PI)
    // If the value isn't fully available in one of our predecessors, then it
    // isn't fully available in this block either.  Undo our previous
    // optimistic assumption and bail out.
    if (!IsValueFullyAvailableInBlock(*PI, FullyAvailableBlocks))
      goto SpeculationFailure;
  
  return true;
  
// SpeculationFailure - If we get here, we found out that this is not, after
// all, a fully-available block.  We have a problem if we speculated on this and
// used the speculation to mark other blocks as available.
SpeculationFailure:
  char &BBVal = FullyAvailableBlocks[BB];
  
  // If we didn't speculate on this, just return with it set to false.
  if (BBVal == 2) {
    BBVal = 0;
    return false;
  }

  // If we did speculate on this value, we could have blocks set to 1 that are
  // incorrect.  Walk the (transitive) successors of this block and mark them as
  // 0 if set to one.
  SmallVector<BasicBlock*, 32> BBWorklist;
  BBWorklist.push_back(BB);
  
  while (!BBWorklist.empty()) {
    BasicBlock *Entry = BBWorklist.pop_back_val();
    // Note that this sets blocks to 0 (unavailable) if they happen to not
    // already be in FullyAvailableBlocks.  This is safe.
    char &EntryVal = FullyAvailableBlocks[Entry];
    if (EntryVal == 0) continue;  // Already unavailable.

    // Mark as unavailable.
    EntryVal = 0;
    
    for (succ_iterator I = succ_begin(Entry), E = succ_end(Entry); I != E; ++I)
      BBWorklist.push_back(*I);
  }
  
  return false;
}

/// processNonLocalLoad - Attempt to eliminate a load whose dependencies are
/// non-local by performing PHI construction.
bool GVN::processNonLocalLoad(LoadInst *LI,
                              SmallVectorImpl<Instruction*> &toErase) {
  // Find the non-local dependencies of the load.
  SmallVector<MemoryDependenceAnalysis::NonLocalDepEntry, 64> Deps; 
  MD->getNonLocalPointerDependency(LI->getOperand(0), true, LI->getParent(),
                                   Deps);
  //DEBUG(cerr << "INVESTIGATING NONLOCAL LOAD: " << Deps.size() << *LI);
  
  // If we had to process more than one hundred blocks to find the
  // dependencies, this load isn't worth worrying about.  Optimizing
  // it will be too expensive.
  if (Deps.size() > 100)
    return false;

  // If we had a phi translation failure, we'll have a single entry which is a
  // clobber in the current block.  Reject this early.
  if (Deps.size() == 1 && Deps[0].second.isClobber())
    return false;
  
  // Filter out useless results (non-locals, etc).  Keep track of the blocks
  // where we have a value available in repl, also keep track of whether we see
  // dependencies that produce an unknown value for the load (such as a call
  // that could potentially clobber the load).
  SmallVector<std::pair<BasicBlock*, Value*>, 16> ValuesPerBlock;
  SmallVector<BasicBlock*, 16> UnavailableBlocks;
  
  for (unsigned i = 0, e = Deps.size(); i != e; ++i) {
    BasicBlock *DepBB = Deps[i].first;
    MemDepResult DepInfo = Deps[i].second;
    
    if (DepInfo.isClobber()) {
      UnavailableBlocks.push_back(DepBB);
      continue;
    }
    
    Instruction *DepInst = DepInfo.getInst();
    
    // Loading the allocation -> undef.
    if (isa<AllocationInst>(DepInst)) {
      ValuesPerBlock.push_back(std::make_pair(DepBB, 
                                              UndefValue::get(LI->getType())));
      continue;
    }
  
    if (StoreInst* S = dyn_cast<StoreInst>(DepInst)) {
      // Reject loads and stores that are to the same address but are of 
      // different types.
      // NOTE: 403.gcc does have this case (e.g. in readonly_fields_p) because
      // of bitfield access, it would be interesting to optimize for it at some
      // point.
      if (S->getOperand(0)->getType() != LI->getType()) {
        UnavailableBlocks.push_back(DepBB);
        continue;
      }
      
      ValuesPerBlock.push_back(std::make_pair(DepBB, S->getOperand(0)));
      
    } else if (LoadInst* LD = dyn_cast<LoadInst>(DepInst)) {
      if (LD->getType() != LI->getType()) {
        UnavailableBlocks.push_back(DepBB);
        continue;
      }
      ValuesPerBlock.push_back(std::make_pair(DepBB, LD));
    } else {
      UnavailableBlocks.push_back(DepBB);
      continue;
    }
  }
  
  // If we have no predecessors that produce a known value for this load, exit
  // early.
  if (ValuesPerBlock.empty()) return false;
  
  // If all of the instructions we depend on produce a known value for this
  // load, then it is fully redundant and we can use PHI insertion to compute
  // its value.  Insert PHIs and remove the fully redundant value now.
  if (UnavailableBlocks.empty()) {
    // Use cached PHI construction information from previous runs
    SmallPtrSet<Instruction*, 4> &p = phiMap[LI->getPointerOperand()];
    // FIXME: What does phiMap do? Are we positive it isn't getting invalidated?
    for (SmallPtrSet<Instruction*, 4>::iterator I = p.begin(), E = p.end();
         I != E; ++I) {
      if ((*I)->getParent() == LI->getParent()) {
        DEBUG(cerr << "GVN REMOVING NONLOCAL LOAD #1: " << *LI);
        LI->replaceAllUsesWith(*I);
        if (isa<PointerType>((*I)->getType()))
          MD->invalidateCachedPointerInfo(*I);
        toErase.push_back(LI);
        NumGVNLoad++;
        return true;
      }
      
      ValuesPerBlock.push_back(std::make_pair((*I)->getParent(), *I));
    }
    
    DEBUG(cerr << "GVN REMOVING NONLOCAL LOAD: " << *LI);
    
    DenseMap<BasicBlock*, Value*> BlockReplValues;
    BlockReplValues.insert(ValuesPerBlock.begin(), ValuesPerBlock.end());
    // Perform PHI construction.
    Value* v = GetValueForBlock(LI->getParent(), LI, BlockReplValues, true);
    LI->replaceAllUsesWith(v);
    
    if (!isa<GlobalValue>(v))
      v->takeName(LI);
    if (isa<PointerType>(v->getType()))
      MD->invalidateCachedPointerInfo(v);
    toErase.push_back(LI);
    NumGVNLoad++;
    return true;
  }
  
  if (!EnablePRE || !EnableLoadPRE)
    return false;

  // Okay, we have *some* definitions of the value.  This means that the value
  // is available in some of our (transitive) predecessors.  Lets think about
  // doing PRE of this load.  This will involve inserting a new load into the
  // predecessor when it's not available.  We could do this in general, but
  // prefer to not increase code size.  As such, we only do this when we know
  // that we only have to insert *one* load (which means we're basically moving
  // the load, not inserting a new one).
  
  // Everything we do here is based on local predecessors of LI's block.  If it
  // only has one predecessor, bail now.
  BasicBlock *LoadBB = LI->getParent();
  if (LoadBB->getSinglePredecessor())
    return false;
  
  // If we have a repl set with LI itself in it, this means we have a loop where
  // at least one of the values is LI.  Since this means that we won't be able
  // to eliminate LI even if we insert uses in the other predecessors, we will
  // end up increasing code size.  Reject this by scanning for LI.
  for (unsigned i = 0, e = ValuesPerBlock.size(); i != e; ++i)
    if (ValuesPerBlock[i].second == LI)
      return false;
  
  // Okay, we have some hope :).  Check to see if the loaded value is fully
  // available in all but one predecessor.
  // FIXME: If we could restructure the CFG, we could make a common pred with
  // all the preds that don't have an available LI and insert a new load into
  // that one block.
  BasicBlock *UnavailablePred = 0;

  DenseMap<BasicBlock*, char> FullyAvailableBlocks;
  for (unsigned i = 0, e = ValuesPerBlock.size(); i != e; ++i)
    FullyAvailableBlocks[ValuesPerBlock[i].first] = true;
  for (unsigned i = 0, e = UnavailableBlocks.size(); i != e; ++i)
    FullyAvailableBlocks[UnavailableBlocks[i]] = false;

  for (pred_iterator PI = pred_begin(LoadBB), E = pred_end(LoadBB);
       PI != E; ++PI) {
    if (IsValueFullyAvailableInBlock(*PI, FullyAvailableBlocks))
      continue;
    
    // If this load is not available in multiple predecessors, reject it.
    if (UnavailablePred && UnavailablePred != *PI)
      return false;
    UnavailablePred = *PI;
  }
  
  assert(UnavailablePred != 0 &&
         "Fully available value should be eliminated above!");
  
  // If the loaded pointer is PHI node defined in this block, do PHI translation
  // to get its value in the predecessor.
  Value *LoadPtr = LI->getOperand(0)->DoPHITranslation(LoadBB, UnavailablePred);
  
  // Make sure the value is live in the predecessor.  If it was defined by a
  // non-PHI instruction in this block, we don't know how to recompute it above.
  if (Instruction *LPInst = dyn_cast<Instruction>(LoadPtr))
    if (!DT->dominates(LPInst->getParent(), UnavailablePred)) {
      DEBUG(cerr << "COULDN'T PRE LOAD BECAUSE PTR IS UNAVAILABLE IN PRED: "
                 << *LPInst << *LI << "\n");
      return false;
    }
  
  // We don't currently handle critical edges :(
  if (UnavailablePred->getTerminator()->getNumSuccessors() != 1) {
    DEBUG(cerr << "COULD NOT PRE LOAD BECAUSE OF CRITICAL EDGE '"
                << UnavailablePred->getName() << "': " << *LI);
    return false;
  }
  
  // Okay, we can eliminate this load by inserting a reload in the predecessor
  // and using PHI construction to get the value in the other predecessors, do
  // it.
  DEBUG(cerr << "GVN REMOVING PRE LOAD: " << *LI);
  
  Value *NewLoad = new LoadInst(LoadPtr, LI->getName()+".pre", false,
                                LI->getAlignment(),
                                UnavailablePred->getTerminator());
  
  DenseMap<BasicBlock*, Value*> BlockReplValues;
  BlockReplValues.insert(ValuesPerBlock.begin(), ValuesPerBlock.end());
  BlockReplValues[UnavailablePred] = NewLoad;
  
  // Perform PHI construction.
  Value* v = GetValueForBlock(LI->getParent(), LI, BlockReplValues, true);
  LI->replaceAllUsesWith(v);
  if (!isa<GlobalValue>(v))
    v->takeName(LI);
  if (isa<PointerType>(v->getType()))
    MD->invalidateCachedPointerInfo(v);
  toErase.push_back(LI);
  NumPRELoad++;
  return true;
}

/// processLoad - Attempt to eliminate a load, first by eliminating it
/// locally, and then attempting non-local elimination if that fails.
bool GVN::processLoad(LoadInst *L, SmallVectorImpl<Instruction*> &toErase) {
  if (L->isVolatile())
    return false;
  
  Value* pointer = L->getPointerOperand();

  // ... to a pointer that has been loaded from before...
  MemDepResult dep = MD->getDependency(L);
  
  // If the value isn't available, don't do anything!
  if (dep.isClobber())
    return false;

  // If it is defined in another block, try harder.
  if (dep.isNonLocal())
    return processNonLocalLoad(L, toErase);

  Instruction *DepInst = dep.getInst();
  if (StoreInst *DepSI = dyn_cast<StoreInst>(DepInst)) {
    // Only forward substitute stores to loads of the same type.
    // FIXME: Could do better!
    if (DepSI->getPointerOperand()->getType() != pointer->getType())
      return false;
    
    // Remove it!
    L->replaceAllUsesWith(DepSI->getOperand(0));
    if (isa<PointerType>(DepSI->getOperand(0)->getType()))
      MD->invalidateCachedPointerInfo(DepSI->getOperand(0));
    toErase.push_back(L);
    NumGVNLoad++;
    return true;
  }

  if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInst)) {
    // Only forward substitute stores to loads of the same type.
    // FIXME: Could do better! load i32 -> load i8 -> truncate on little endian.
    if (DepLI->getType() != L->getType())
      return false;
    
    // Remove it!
    L->replaceAllUsesWith(DepLI);
    if (isa<PointerType>(DepLI->getType()))
      MD->invalidateCachedPointerInfo(DepLI);
    toErase.push_back(L);
    NumGVNLoad++;
    return true;
  }
  
  // If this load really doesn't depend on anything, then we must be loading an
  // undef value.  This can happen when loading for a fresh allocation with no
  // intervening stores, for example.
  if (isa<AllocationInst>(DepInst)) {
    L->replaceAllUsesWith(UndefValue::get(L->getType()));
    toErase.push_back(L);
    NumGVNLoad++;
    return true;
  }

  return false;
}

Value* GVN::lookupNumber(BasicBlock* BB, uint32_t num) {
  DenseMap<BasicBlock*, ValueNumberScope*>::iterator I = localAvail.find(BB);
  if (I == localAvail.end())
    return 0;
  
  ValueNumberScope* locals = I->second;
  
  while (locals) {
    DenseMap<uint32_t, Value*>::iterator I = locals->table.find(num);
    if (I != locals->table.end())
      return I->second;
    else
      locals = locals->parent;
  }
  
  return 0;
}

/// AttemptRedundancyElimination - If the "fast path" of redundancy elimination
/// by inheritance from the dominator fails, see if we can perform phi 
/// construction to eliminate the redundancy.
Value* GVN::AttemptRedundancyElimination(Instruction* orig, unsigned valno) {
  BasicBlock* BaseBlock = orig->getParent();
  
  SmallPtrSet<BasicBlock*, 4> Visited;
  SmallVector<BasicBlock*, 8> Stack;
  Stack.push_back(BaseBlock);
  
  DenseMap<BasicBlock*, Value*> Results;
  
  // Walk backwards through our predecessors, looking for instances of the
  // value number we're looking for.  Instances are recorded in the Results
  // map, which is then used to perform phi construction.
  while (!Stack.empty()) {
    BasicBlock* Current = Stack.back();
    Stack.pop_back();
    
    // If we've walked all the way to a proper dominator, then give up. Cases
    // where the instance is in the dominator will have been caught by the fast
    // path, and any cases that require phi construction further than this are
    // probably not worth it anyways.  Note that this is a SIGNIFICANT compile
    // time improvement.
    if (DT->properlyDominates(Current, orig->getParent())) return 0;
    
    DenseMap<BasicBlock*, ValueNumberScope*>::iterator LA =
                                                       localAvail.find(Current);
    if (LA == localAvail.end()) return 0;
    DenseMap<uint32_t, Value*>::iterator V = LA->second->table.find(valno);
    
    if (V != LA->second->table.end()) {
      // Found an instance, record it.
      Results.insert(std::make_pair(Current, V->second));
      continue;
    }
    
    // If we reach the beginning of the function, then give up.
    if (pred_begin(Current) == pred_end(Current))
      return 0;
    
    for (pred_iterator PI = pred_begin(Current), PE = pred_end(Current);
         PI != PE; ++PI)
      if (Visited.insert(*PI))
        Stack.push_back(*PI);
  }
  
  // If we didn't find instances, give up.  Otherwise, perform phi construction.
  if (Results.size() == 0)
    return 0;
  else
    return GetValueForBlock(BaseBlock, orig, Results, true);
}

/// processInstruction - When calculating availability, handle an instruction
/// by inserting it into the appropriate sets
bool GVN::processInstruction(Instruction *I,
                             SmallVectorImpl<Instruction*> &toErase) {
  if (LoadInst* L = dyn_cast<LoadInst>(I)) {
    bool changed = processLoad(L, toErase);
    
    if (!changed) {
      unsigned num = VN.lookup_or_add(L);
      localAvail[I->getParent()]->table.insert(std::make_pair(num, L));
    }
    
    return changed;
  }
  
  uint32_t nextNum = VN.getNextUnusedValueNumber();
  unsigned num = VN.lookup_or_add(I);
  
  // Allocations are always uniquely numbered, so we can save time and memory
  // by fast failing them.
  if (isa<AllocationInst>(I) || isa<TerminatorInst>(I)) {
    localAvail[I->getParent()]->table.insert(std::make_pair(num, I));
    return false;
  }
  
  // Collapse PHI nodes
  if (PHINode* p = dyn_cast<PHINode>(I)) {
    Value* constVal = CollapsePhi(p);
    
    if (constVal) {
      for (PhiMapType::iterator PI = phiMap.begin(), PE = phiMap.end();
           PI != PE; ++PI)
        PI->second.erase(p);
        
      p->replaceAllUsesWith(constVal);
      if (isa<PointerType>(constVal->getType()))
        MD->invalidateCachedPointerInfo(constVal);
      VN.erase(p);
      
      toErase.push_back(p);
    } else {
      localAvail[I->getParent()]->table.insert(std::make_pair(num, I));
    }
  
  // If the number we were assigned was a brand new VN, then we don't
  // need to do a lookup to see if the number already exists
  // somewhere in the domtree: it can't!
  } else if (num == nextNum) {
    localAvail[I->getParent()]->table.insert(std::make_pair(num, I));
    
  // Perform fast-path value-number based elimination of values inherited from
  // dominators.
  } else if (Value* repl = lookupNumber(I->getParent(), num)) {
    // Remove it!
    VN.erase(I);
    I->replaceAllUsesWith(repl);
    if (isa<PointerType>(repl->getType()))
      MD->invalidateCachedPointerInfo(repl);
    toErase.push_back(I);
    return true;

#if 0
  // Perform slow-pathvalue-number based elimination with phi construction.
  } else if (Value* repl = AttemptRedundancyElimination(I, num)) {
    // Remove it!
    VN.erase(I);
    I->replaceAllUsesWith(repl);
    if (isa<PointerType>(repl->getType()))
      MD->invalidateCachedPointerInfo(repl);
    toErase.push_back(I);
    return true;
#endif
  } else {
    localAvail[I->getParent()]->table.insert(std::make_pair(num, I));
  }
  
  return false;
}

/// runOnFunction - This is the main transformation entry point for a function.
bool GVN::runOnFunction(Function& F) {
  MD = &getAnalysis<MemoryDependenceAnalysis>();
  DT = &getAnalysis<DominatorTree>();
  VN.setAliasAnalysis(&getAnalysis<AliasAnalysis>());
  VN.setMemDep(MD);
  VN.setDomTree(DT);
  
  bool changed = false;
  bool shouldContinue = true;
  
  // Merge unconditional branches, allowing PRE to catch more
  // optimization opportunities.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ) {
    BasicBlock* BB = FI;
    ++FI;
    bool removedBlock = MergeBlockIntoPredecessor(BB, this);
    if (removedBlock) NumGVNBlocks++;
    
    changed |= removedBlock;
  }
  
  unsigned Iteration = 0;
  
  while (shouldContinue) {
    DEBUG(cerr << "GVN iteration: " << Iteration << "\n");
    shouldContinue = iterateOnFunction(F);
    changed |= shouldContinue;
    ++Iteration;
  }
  
  if (EnablePRE) {
    bool PREChanged = true;
    while (PREChanged) {
      PREChanged = performPRE(F);
      changed |= PREChanged;
    }
  }
  // FIXME: Should perform GVN again after PRE does something.  PRE can move
  // computations into blocks where they become fully redundant.  Note that
  // we can't do this until PRE's critical edge splitting updates memdep.
  // Actually, when this happens, we should just fully integrate PRE into GVN.

  cleanupGlobalSets();

  return changed;
}


bool GVN::processBlock(BasicBlock* BB) {
  DomTreeNode* DTN = DT->getNode(BB);
  // FIXME: Kill off toErase by doing erasing eagerly in a helper function (and
  // incrementing BI before processing an instruction).
  SmallVector<Instruction*, 8> toErase;
  bool changed_function = false;
  
  if (DTN->getIDom())
    localAvail[BB] =
                  new ValueNumberScope(localAvail[DTN->getIDom()->getBlock()]);
  else
    localAvail[BB] = new ValueNumberScope(0);
  
  for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
       BI != BE;) {
    changed_function |= processInstruction(BI, toErase);
    if (toErase.empty()) {
      ++BI;
      continue;
    }
    
    // If we need some instructions deleted, do it now.
    NumGVNInstr += toErase.size();
    
    // Avoid iterator invalidation.
    bool AtStart = BI == BB->begin();
    if (!AtStart)
      --BI;

    for (SmallVector<Instruction*, 4>::iterator I = toErase.begin(),
         E = toErase.end(); I != E; ++I) {
      DEBUG(cerr << "GVN removed: " << **I);
      MD->removeInstruction(*I);
      (*I)->eraseFromParent();
      DEBUG(verifyRemoved(*I));
    }
    toErase.clear();

    if (AtStart)
      BI = BB->begin();
    else
      ++BI;
  }
  
  return changed_function;
}

/// performPRE - Perform a purely local form of PRE that looks for diamond
/// control flow patterns and attempts to perform simple PRE at the join point.
bool GVN::performPRE(Function& F) {
  bool Changed = false;
  SmallVector<std::pair<TerminatorInst*, unsigned>, 4> toSplit;
  DenseMap<BasicBlock*, Value*> predMap;
  for (df_iterator<BasicBlock*> DI = df_begin(&F.getEntryBlock()),
       DE = df_end(&F.getEntryBlock()); DI != DE; ++DI) {
    BasicBlock* CurrentBlock = *DI;
    
    // Nothing to PRE in the entry block.
    if (CurrentBlock == &F.getEntryBlock()) continue;
    
    for (BasicBlock::iterator BI = CurrentBlock->begin(),
         BE = CurrentBlock->end(); BI != BE; ) {
      Instruction *CurInst = BI++;
      
      if (isa<AllocationInst>(CurInst) || isa<TerminatorInst>(CurInst) ||
          isa<PHINode>(CurInst) || CurInst->mayReadFromMemory() ||
          CurInst->mayWriteToMemory())
        continue;
      
      uint32_t valno = VN.lookup(CurInst);
      
      // Look for the predecessors for PRE opportunities.  We're
      // only trying to solve the basic diamond case, where
      // a value is computed in the successor and one predecessor,
      // but not the other.  We also explicitly disallow cases
      // where the successor is its own predecessor, because they're
      // more complicated to get right.
      unsigned numWith = 0;
      unsigned numWithout = 0;
      BasicBlock* PREPred = 0;
      predMap.clear();

      for (pred_iterator PI = pred_begin(CurrentBlock),
           PE = pred_end(CurrentBlock); PI != PE; ++PI) {
        // We're not interested in PRE where the block is its
        // own predecessor, on in blocks with predecessors
        // that are not reachable.
        if (*PI == CurrentBlock) {
          numWithout = 2;
          break;
        } else if (!localAvail.count(*PI))  {
          numWithout = 2;
          break;
        }
        
        DenseMap<uint32_t, Value*>::iterator predV = 
                                            localAvail[*PI]->table.find(valno);
        if (predV == localAvail[*PI]->table.end()) {
          PREPred = *PI;
          numWithout++;
        } else if (predV->second == CurInst) {
          numWithout = 2;
        } else {
          predMap[*PI] = predV->second;
          numWith++;
        }
      }
      
      // Don't do PRE when it might increase code size, i.e. when
      // we would need to insert instructions in more than one pred.
      if (numWithout != 1 || numWith == 0)
        continue;
      
      // We can't do PRE safely on a critical edge, so instead we schedule
      // the edge to be split and perform the PRE the next time we iterate
      // on the function.
      unsigned succNum = 0;
      for (unsigned i = 0, e = PREPred->getTerminator()->getNumSuccessors();
           i != e; ++i)
        if (PREPred->getTerminator()->getSuccessor(i) == CurrentBlock) {
          succNum = i;
          break;
        }
        
      if (isCriticalEdge(PREPred->getTerminator(), succNum)) {
        toSplit.push_back(std::make_pair(PREPred->getTerminator(), succNum));
        continue;
      }
      
      // Instantiate the expression the in predecessor that lacked it.
      // Because we are going top-down through the block, all value numbers
      // will be available in the predecessor by the time we need them.  Any
      // that weren't original present will have been instantiated earlier
      // in this loop.
      Instruction* PREInstr = CurInst->clone();
      bool success = true;
      for (unsigned i = 0, e = CurInst->getNumOperands(); i != e; ++i) {
        Value *Op = PREInstr->getOperand(i);
        if (isa<Argument>(Op) || isa<Constant>(Op) || isa<GlobalValue>(Op))
          continue;
        
        if (Value *V = lookupNumber(PREPred, VN.lookup(Op))) {
          PREInstr->setOperand(i, V);
        } else {
          success = false;
          break;
        }
      }
      
      // Fail out if we encounter an operand that is not available in
      // the PRE predecessor.  This is typically because of loads which 
      // are not value numbered precisely.
      if (!success) {
        delete PREInstr;
        DEBUG(verifyRemoved(PREInstr));
        continue;
      }
      
      PREInstr->insertBefore(PREPred->getTerminator());
      PREInstr->setName(CurInst->getName() + ".pre");
      predMap[PREPred] = PREInstr;
      VN.add(PREInstr, valno);
      NumGVNPRE++;
      
      // Update the availability map to include the new instruction.
      localAvail[PREPred]->table.insert(std::make_pair(valno, PREInstr));
      
      // Create a PHI to make the value available in this block.
      PHINode* Phi = PHINode::Create(CurInst->getType(),
                                     CurInst->getName() + ".pre-phi",
                                     CurrentBlock->begin());
      for (pred_iterator PI = pred_begin(CurrentBlock),
           PE = pred_end(CurrentBlock); PI != PE; ++PI)
        Phi->addIncoming(predMap[*PI], *PI);
      
      VN.add(Phi, valno);
      localAvail[CurrentBlock]->table[valno] = Phi;
      
      CurInst->replaceAllUsesWith(Phi);
      if (isa<PointerType>(Phi->getType()))
        MD->invalidateCachedPointerInfo(Phi);
      VN.erase(CurInst);
      
      DEBUG(cerr << "GVN PRE removed: " << *CurInst);
      MD->removeInstruction(CurInst);
      CurInst->eraseFromParent();
      DEBUG(verifyRemoved(CurInst));
      Changed = true;
    }
  }
  
  for (SmallVector<std::pair<TerminatorInst*, unsigned>, 4>::iterator
       I = toSplit.begin(), E = toSplit.end(); I != E; ++I)
    SplitCriticalEdge(I->first, I->second, this);
  
  return Changed || toSplit.size();
}

/// iterateOnFunction - Executes one iteration of GVN
bool GVN::iterateOnFunction(Function &F) {
  cleanupGlobalSets();

  // Top-down walk of the dominator tree
  bool changed = false;
#if 0
  // Needed for value numbering with phi construction to work.
  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator RI = RPOT.begin(),
       RE = RPOT.end(); RI != RE; ++RI)
    changed |= processBlock(*RI);
#else
  for (df_iterator<DomTreeNode*> DI = df_begin(DT->getRootNode()),
       DE = df_end(DT->getRootNode()); DI != DE; ++DI)
    changed |= processBlock(DI->getBlock());
#endif

  return changed;
}

void GVN::cleanupGlobalSets() {
  VN.clear();
  phiMap.clear();

  for (DenseMap<BasicBlock*, ValueNumberScope*>::iterator
       I = localAvail.begin(), E = localAvail.end(); I != E; ++I)
    delete I->second;
  localAvail.clear();
}

/// verifyRemoved - Verify that the specified instruction does not occur in our
/// internal data structures.
void GVN::verifyRemoved(const Instruction *Inst) const {
  VN.verifyRemoved(Inst);

  // Walk through the PHI map to make sure the instruction isn't hiding in there
  // somewhere.
  for (PhiMapType::iterator
         I = phiMap.begin(), E = phiMap.end(); I != E; ++I) {
    assert(I->first != Inst && "Inst is still a key in PHI map!");

    for (SmallPtrSet<Instruction*, 4>::iterator
           II = I->second.begin(), IE = I->second.end(); II != IE; ++II) {
      assert(*II != Inst && "Inst is still a value in PHI map!");
    }
  }

  // Walk through the value number scope to make sure the instruction isn't
  // ferreted away in it.
  for (DenseMap<BasicBlock*, ValueNumberScope*>::iterator
         I = localAvail.begin(), E = localAvail.end(); I != E; ++I) {
    const ValueNumberScope *VNS = I->second;

    while (VNS) {
      for (DenseMap<uint32_t, Value*>::iterator
             II = VNS->table.begin(), IE = VNS->table.end(); II != IE; ++II) {
        assert(II->second != Inst && "Inst still in value numbering scope!");
      }

      VNS = VNS->parent;
    }
  }
}
