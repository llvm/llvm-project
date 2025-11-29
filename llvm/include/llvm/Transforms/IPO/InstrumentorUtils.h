//===- Transforms/IPO/InstrumentorUtils.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// General utilities for the Instrumentor pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INSTRUMENTOR_UTILS_H
#define LLVM_TRANSFORMS_IPO_INSTRUMENTOR_UTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <bitset>
#include <tuple>

namespace llvm {
namespace instrumentor {

/// An IR builder augmented with extra information for the instrumentor pass.
/// The underlying IR builder features an insertion callback to keep track of
/// the new instructions.
struct InstrumentorIRBuilderTy {
  /// Construct an IR builder for the module \p M.
  InstrumentorIRBuilderTy(Module &M)
      : M(M), Ctx(M.getContext()),
        IRB(Ctx, ConstantFolder(),
            // Save the inserted instructions in a structure.
            IRBuilderCallbackInserter(
                [&](Instruction *I) { NewInsts[I] = Epoch; })) {}

  /// Destroy the IR builder and remove all erasable instructions cached during
  /// the process of instrumenting.
  ~InstrumentorIRBuilderTy() {
    for (auto *I : ErasableInstructions) {
      if (!I->getType()->isVoidTy())
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  /// Get a temporary alloca to communicate (large) values with the runtime.
  AllocaInst *getAlloca(Function *Fn, Type *Ty, bool MatchType = false) {
    const DataLayout &DL = Fn->getDataLayout();
    auto *&AllocaList = AllocaMap[{Fn, DL.getTypeAllocSize(Ty)}];
    if (!AllocaList)
      AllocaList = new AllocaListTy;
    AllocaInst *AI = nullptr;
    for (auto *&ListAI : *AllocaList) {
      if (MatchType && ListAI->getAllocatedType() != Ty)
        continue;
      AI = ListAI;
      ListAI = *AllocaList->rbegin();
      break;
    }
    if (AI)
      AllocaList->pop_back();
    else
      AI = new AllocaInst(Ty, DL.getAllocaAddrSpace(), "",
                          Fn->getEntryBlock().begin());
    UsedAllocas[AI] = AllocaList;
    return AI;
  }

  /// Return the temporary allocas.
  void returnAllocas() {
    for (auto [AI, List] : UsedAllocas)
      List->push_back(AI);
    UsedAllocas.clear();
  }

  /// Save instruction \p I to be erased later. The instructions are erased when
  /// the IR builder is destroyed.
  void eraseLater(Instruction *I) { ErasableInstructions.insert(I); }

  /// Commonly used values for IR inspection and creation.
  ///{
  Module &M;

  LLVMContext &Ctx;

  const DataLayout &DL = M.getDataLayout();

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);
  Constant *NullPtrVal = Constant::getNullValue(PtrTy);
  ///}

  using AllocaListTy = SmallVector<AllocaInst *>;

  /// Map that holds a list of currently available allocas for a function and
  /// alloca size.
  DenseMap<std::pair<Function *, unsigned>, AllocaListTy *> AllocaMap;

  /// Map that holds the currently used allocas and the list where they belong.
  /// Once an alloca has to be returned, it is returned directly to its list.
  DenseMap<AllocaInst *, AllocaListTy *> UsedAllocas;

  /// Instructions that should be erased later.
  SmallPtrSet<Instruction *, 32> ErasableInstructions;

  /// The underlying IR builder with insertion callback.
  IRBuilder<ConstantFolder, IRBuilderCallbackInserter> IRB;

  /// The current epoch number. Each instrumentation, e.g., of an instruction,
  /// is happening in a dedicated epoch. The epoch allows to determine if
  /// instrumentation instructions were already around, due to prior
  /// instrumentations, or have been introduced to support the current
  /// instrumentation, e.g., compute information about the current instruction.
  unsigned Epoch = 0;

  /// A mapping from instrumentation instructions to the epoch they have been
  /// created.
  DenseMap<Instruction *, unsigned> NewInsts;
};

/// Helper that represent the caches for instrumentation call arguments. The
/// value of an argument may not need to be recomputed between the pre and post
/// instrumentation calls.
struct InstrumentationCaches {
  /// A cache for direct and indirect arguments. The cache is indexed by the
  /// epoch, the instrumentation opportunity name and the argument name. The
  /// result is a value.
  DenseMap<std::tuple<unsigned, StringRef, StringRef>, Value *> DirectArgCache;
  DenseMap<std::tuple<unsigned, StringRef, StringRef>, Value *>
      IndirectArgCache;
};

/// Boolean option bitset with a compile-time number of bits to store as many
/// options as the enumeration type \p EnumTy defines. The enumeration type is
/// expected to have an ascending and consecutive values, starting at zero, and
/// the last value being artificial and named as NumConfig (i.e., the number of
/// values in the enumeration).
template <typename EnumTy> struct BaseConfigTy {
  /// The bistset with as many bits as the enumeration's values.
  std::bitset<static_cast<int>(EnumTy::NumConfig)> Options;

  /// Construct the option bitset with all bits set to \p Enable. If not
  /// provided, all options are enabled.
  BaseConfigTy(bool Enable = true) {
    if (Enable)
      Options.set();
  }

  /// Check if the option \p Opt is enabled.
  bool has(EnumTy Opt) const { return Options.test(static_cast<int>(Opt)); }

  /// Set the boolean value of option \p Opt to \p Value.
  void set(EnumTy Opt, bool Value = true) {
    Options.set(static_cast<int>(Opt), Value);
  }
};

} // namespace instrumentor
} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INSTRUMENTOR_UTILS_H
