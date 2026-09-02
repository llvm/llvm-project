//===- GVNValueTable.h - Value table for GVN ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides a data structure for mapping values and expressions to
/// congruence class IDs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVNVALUETABLE_H
#define LLVM_TRANSFORMS_SCALAR_GVNVALUETABLE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Scalar/GVNValueTable.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {

class AAResults;
class AssumeInst;
class AssumptionCache;
class BasicBlock;
class BatchAAResults;
class CallInst;
class CondBrInst;
class EarliestEscapeAnalysis;
class ExtractValueInst;
class Function;
class FunctionPass;
class GVNLegacyPass;
class GVNPass;
class GetElementPtrInst;
class ImplicitControlFlowTracking;
class LoadInst;
class LoopInfo;
class MemDepResult;
class MemoryAccess;
class MemoryDependenceResults;
class MemoryLocation;
class MemorySSA;
class MemorySSAUpdater;
class NonLocalDepResult;
class OptimizationRemarkEmitter;
class PHINode;
class TargetLibraryInfo;
class Value;
class IntrinsicInst;

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.
class GVNValueTable {
public:
  struct Expression;

private:
  DenseMap<Value *, uint32_t> ValueNumbering;
  DenseMap<Expression, uint32_t> ExpressionNumbering;

  // Expressions is the vector of Expression. ExprIdx is the mapping from
  // value number to the index of Expression in Expressions. We use it
  // instead of a DenseMap because filling such mapping is faster than
  // filling a DenseMap and the compile time is a little better.
  uint32_t NextExprNumber = 0;

  std::vector<Expression> Expressions;
  std::vector<uint32_t> ExprIdx;

  // Value number to PHINode mapping. Used for phi-translate in scalarpre.
  DenseMap<uint32_t, PHINode *> NumberingPhi;

  // Value number to BasicBlock mapping. Used for phi-translate across
  // MemoryPhis.
  DenseMap<uint32_t, BasicBlock *> NumberingBB;

  // Cache for phi-translate in scalarpre.
  using PhiTranslateMap =
      DenseMap<std::pair<uint32_t, const BasicBlock *>, uint32_t>;
  PhiTranslateMap PhiTranslateTable;

  AAResults *AA = nullptr;
  MemoryDependenceResults *MD = nullptr;
  bool IsMDEnabled = false;
  MemorySSA *MSSA = nullptr;
  bool IsMSSAEnabled = false;
  DominatorTree *DT = nullptr;

  uint32_t NextValueNumber = 1;

  Expression createExpr(Instruction *I);
  Expression createCmpExpr(unsigned Opcode, CmpInst::Predicate Predicate,
                           Value *LHS, Value *RHS);
  Expression createExtractvalueExpr(ExtractValueInst *EI);
  Expression createGEPExpr(GetElementPtrInst *GEP);
  uint32_t lookupOrAddCall(CallInst *C);
  uint32_t computeLoadStoreVN(Instruction *I);
  uint32_t phiTranslateImpl(const BasicBlock *BB, const BasicBlock *PhiBlock,
                            uint32_t Num, GVNPass &GVN);
  bool areCallValsEqual(uint32_t Num, uint32_t NewNum, const BasicBlock *Pred,
                        const BasicBlock *PhiBlock, GVNPass &GVN);
  std::pair<uint32_t, bool> assignExpNewValueNum(Expression &Exp);
  bool areAllValsInBB(uint32_t Num, const BasicBlock *BB, GVNPass &GVN);
  void addMemoryStateToExp(Instruction *I, Expression &Exp);

public:
  LLVM_ABI GVNValueTable();
  LLVM_ABI GVNValueTable(const GVNValueTable &Arg);
  LLVM_ABI GVNValueTable(GVNValueTable &&Arg);
  LLVM_ABI ~GVNValueTable();
  LLVM_ABI GVNValueTable &operator=(const GVNValueTable &Arg);

  LLVM_ABI void add(Value *V, uint32_t Num);
  LLVM_ABI uint32_t lookupOrAdd(MemoryAccess *MA);
  LLVM_ABI uint32_t lookupOrAdd(Value *V);
  LLVM_ABI uint32_t lookup(Value *V, bool Verify = true) const;
  LLVM_ABI uint32_t lookupOrAddCmp(unsigned Opcode, CmpInst::Predicate Pred,
                                   Value *LHS, Value *RHS);
  LLVM_ABI uint32_t lookupPtrToInt(Value *Ptr, Type *Ty);
  LLVM_ABI uint32_t phiTranslate(const BasicBlock *BB,
                                 const BasicBlock *PhiBlock, uint32_t Num,
                                 GVNPass &GVN);
  LLVM_ABI void eraseTranslateCacheEntry(uint32_t Num,
                                         const BasicBlock &CurrBlock);
  LLVM_ABI bool exists(Value *V) const;
  LLVM_ABI void clear();
  LLVM_ABI void erase(Value *V);
  void setAliasAnalysis(AAResults *A) { AA = A; }
  AAResults *getAliasAnalysis() const { return AA; }
  void setMemDep(MemoryDependenceResults *M, bool MDEnabled = true) {
    MD = M;
    IsMDEnabled = MDEnabled;
  }
  void setMemorySSA(MemorySSA *M, bool MSSAEnabled = false) {
    MSSA = M;
    IsMSSAEnabled = MSSAEnabled;
  }
  void setDomTree(DominatorTree *D) { DT = D; }
  uint32_t getNextUnusedValueNumber() { return NextValueNumber; }
  LLVM_ABI void verifyRemoved(const Value *) const;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVNVALUETABLE_H
