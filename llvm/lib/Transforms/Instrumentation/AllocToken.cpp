//===- AllocToken.cpp - Allocation token instrumentation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements AllocToken, an instrumentation pass that
// replaces allocation calls with token-enabled versions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/AllocToken.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/SipHash.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

using namespace llvm;

#define DEBUG_TYPE "alloc-token"

namespace {

//===--- Constants --------------------------------------------------------===//

enum class TokenMode : unsigned {
  /// Incrementally increasing token ID.
  Increment = 0,

  /// Simple mode that returns a statically-assigned random token ID.
  Random = 1,

  /// Token ID based on allocated type hash.
  TypeHash = 2,

  /// Token ID based on allocated type hash, where the top half ID-space is
  /// reserved for types that contain pointers and the bottom half for types
  /// that do not contain pointers.
  TypeHashPointerSplit = 3,
};

//===--- Command-line options ---------------------------------------------===//

cl::opt<TokenMode> ClMode(
    "alloc-token-mode", cl::Hidden, cl::desc("Token assignment mode"),
    cl::init(TokenMode::TypeHashPointerSplit),
    cl::values(
        clEnumValN(TokenMode::Increment, "increment",
                   "Incrementally increasing token ID"),
        clEnumValN(TokenMode::Random, "random",
                   "Statically-assigned random token ID"),
        clEnumValN(TokenMode::TypeHash, "typehash",
                   "Token ID based on allocated type hash"),
        clEnumValN(
            TokenMode::TypeHashPointerSplit, "typehashpointersplit",
            "Token ID based on allocated type hash, where the top half "
            "ID-space is reserved for types that contain pointers and the "
            "bottom half for types that do not contain pointers. ")));

cl::opt<std::string> ClFuncPrefix("alloc-token-prefix",
                                  cl::desc("The allocation function prefix"),
                                  cl::Hidden, cl::init("__alloc_token_"));

cl::opt<uint64_t> ClMaxTokens("alloc-token-max",
                              cl::desc("Maximum number of tokens (0 = no max)"),
                              cl::Hidden, cl::init(0));

cl::opt<bool>
    ClFastABI("alloc-token-fast-abi",
              cl::desc("The token ID is encoded in the function name"),
              cl::Hidden, cl::init(false));

// Instrument libcalls only by default - compatible allocators only need to take
// care of providing standard allocation functions. With extended coverage, also
// instrument non-libcall allocation function calls with !alloc_token
// metadata.
cl::opt<bool>
    ClExtended("alloc-token-extended",
               cl::desc("Extend coverage to custom allocation functions"),
               cl::Hidden, cl::init(false));

// C++ defines ::operator new (and variants) as replaceable (vs. standard
// library versions), which are nobuiltin, and are therefore not covered by
// isAllocationFn(). Cover by default, as users of AllocToken are already
// required to provide token-aware allocation functions (no defaults).
cl::opt<bool> ClCoverReplaceableNew("alloc-token-cover-replaceable-new",
                                    cl::desc("Cover replaceable operator new"),
                                    cl::Hidden, cl::init(true));

cl::opt<uint64_t> ClFallbackToken(
    "alloc-token-fallback",
    cl::desc("The default fallback token where none could be determined"),
    cl::Hidden, cl::init(0));

//===--- Statistics -------------------------------------------------------===//

STATISTIC(NumFunctionsInstrumented, "Functions instrumented");
STATISTIC(NumAllocationsInstrumented, "Allocations instrumented");

//===----------------------------------------------------------------------===//

/// Returns the !alloc_token metadata if available.
///
/// Expected format is: !{<type-name>, <contains-pointer>}
MDNode *getAllocTokenMetadata(const CallBase &CB) {
  MDNode *Ret = CB.getMetadata(LLVMContext::MD_alloc_token);
  if (!Ret)
    return nullptr;
  assert(Ret->getNumOperands() == 2 && "bad !alloc_token");
  assert(isa<MDString>(Ret->getOperand(0)));
  assert(isa<ConstantAsMetadata>(Ret->getOperand(1)));
  return Ret;
}

bool containsPointer(const MDNode *MD) {
  ConstantAsMetadata *C = cast<ConstantAsMetadata>(MD->getOperand(1));
  auto *CI = cast<ConstantInt>(C->getValue());
  return CI->getValue().getBoolValue();
}

class ModeBase {
public:
  explicit ModeBase(const IntegerType &TokenTy, uint64_t MaxTokens)
      : MaxTokens(MaxTokens ? MaxTokens : TokenTy.getBitMask()) {
    assert(MaxTokens <= TokenTy.getBitMask());
  }

protected:
  uint64_t boundedToken(uint64_t Val) const {
    assert(MaxTokens != 0);
    return Val % MaxTokens;
  }

  const uint64_t MaxTokens;
};

/// Implementation for TokenMode::Increment.
class IncrementMode : public ModeBase {
public:
  using ModeBase::ModeBase;

  uint64_t operator()(const CallBase &CB, OptimizationRemarkEmitter &) {
    return boundedToken(Counter++);
  }

private:
  uint64_t Counter = 0;
};

/// Implementation for TokenMode::Random.
class RandomMode : public ModeBase {
public:
  RandomMode(const IntegerType &TokenTy, uint64_t MaxTokens,
             std::unique_ptr<RandomNumberGenerator> RNG)
      : ModeBase(TokenTy, MaxTokens), RNG(std::move(RNG)) {}
  uint64_t operator()(const CallBase &CB, OptimizationRemarkEmitter &) {
    return boundedToken((*RNG)());
  }

private:
  std::unique_ptr<RandomNumberGenerator> RNG;
};

/// Implementation for TokenMode::TypeHash. The implementation ensures
/// hashes are stable across different compiler invocations. Uses SipHash as the
/// hash function.
class TypeHashMode : public ModeBase {
public:
  using ModeBase::ModeBase;

  uint64_t operator()(const CallBase &CB, OptimizationRemarkEmitter &ORE) {
    const auto [N, H] = getHash(CB, ORE);
    return N ? boundedToken(H) : H;
  }

protected:
  std::pair<MDNode *, uint64_t> getHash(const CallBase &CB,
                                        OptimizationRemarkEmitter &ORE) {
    if (MDNode *N = getAllocTokenMetadata(CB)) {
      MDString *S = cast<MDString>(N->getOperand(0));
      return {N, getStableSipHash(S->getString())};
    }
    // Fallback.
    remarkNoMetadata(CB, ORE);
    return {nullptr, ClFallbackToken};
  }

  /// Remark that there was no precise type information.
  static void remarkNoMetadata(const CallBase &CB,
                               OptimizationRemarkEmitter &ORE) {
    ORE.emit([&] {
      ore::NV FuncNV("Function", CB.getParent()->getParent());
      const Function *Callee = CB.getCalledFunction();
      ore::NV CalleeNV("Callee", Callee ? Callee->getName() : "<unknown>");
      return OptimizationRemark(DEBUG_TYPE, "NoAllocToken", &CB)
             << "Call to '" << CalleeNV << "' in '" << FuncNV
             << "' without source-level type token";
    });
  }
};

/// Implementation for TokenMode::TypeHashPointerSplit.
class TypeHashPointerSplitMode : public TypeHashMode {
public:
  using TypeHashMode::TypeHashMode;

  uint64_t operator()(const CallBase &CB, OptimizationRemarkEmitter &ORE) {
    if (MaxTokens == 1)
      return 0;
    const uint64_t HalfTokens = MaxTokens / 2;
    const auto [N, H] = getHash(CB, ORE);
    if (!N) {
      // Pick the fallback token (ClFallbackToken), which by default is 0,
      // meaning it'll fall into the pointer-less bucket. Override by setting
      // -alloc-token-fallback if that is the wrong choice.
      return H;
    }
    uint64_t Hash = H % HalfTokens; // base hash
    if (containsPointer(N))
      Hash += HalfTokens;
    return Hash;
  }
};

// Apply opt overrides.
AllocTokenOptions transformOptionsFromCl(AllocTokenOptions Opts) {
  if (!Opts.MaxTokens.has_value())
    Opts.MaxTokens = ClMaxTokens;
  Opts.FastABI |= ClFastABI;
  Opts.Extended |= ClExtended;
  return Opts;
}

class AllocToken {
public:
  explicit AllocToken(AllocTokenOptions Opts, Module &M,
                      ModuleAnalysisManager &MAM)
      : Options(transformOptionsFromCl(std::move(Opts))), Mod(M),
        FAM(MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
        Mode(IncrementMode(*IntPtrTy, *Options.MaxTokens)) {
    switch (ClMode.getValue()) {
    case TokenMode::Increment:
      break;
    case TokenMode::Random:
      Mode.emplace<RandomMode>(*IntPtrTy, *Options.MaxTokens,
                               M.createRNG(DEBUG_TYPE));
      break;
    case TokenMode::TypeHash:
      Mode.emplace<TypeHashMode>(*IntPtrTy, *Options.MaxTokens);
      break;
    case TokenMode::TypeHashPointerSplit:
      Mode.emplace<TypeHashPointerSplitMode>(*IntPtrTy, *Options.MaxTokens);
      break;
    }
  }

  bool instrumentFunction(Function &F);

private:
  /// Returns the LibFunc (or NotLibFunc) if this call should be instrumented.
  std::optional<LibFunc>
  shouldInstrumentCall(const CallBase &CB, const TargetLibraryInfo &TLI) const;

  /// Returns true for functions that are eligible for instrumentation.
  static bool isInstrumentableLibFunc(LibFunc Func, const CallBase &CB,
                                      const TargetLibraryInfo &TLI);

  /// Returns true for isAllocationFn() functions that we should ignore.
  static bool ignoreInstrumentableLibFunc(LibFunc Func);

  /// Replace a call/invoke with a call/invoke to the allocation function
  /// with token ID.
  bool replaceAllocationCall(CallBase *CB, LibFunc Func,
                             OptimizationRemarkEmitter &ORE,
                             const TargetLibraryInfo &TLI);

  /// Return replacement function for a LibFunc that takes a token ID.
  FunctionCallee getTokenAllocFunction(const CallBase &CB, uint64_t TokenID,
                                       LibFunc OriginalFunc);

  /// Return the token ID from metadata in the call.
  uint64_t getToken(const CallBase &CB, OptimizationRemarkEmitter &ORE) {
    return std::visit([&](auto &&Mode) { return Mode(CB, ORE); }, Mode);
  }

  const AllocTokenOptions Options;
  Module &Mod;
  IntegerType *IntPtrTy = Mod.getDataLayout().getIntPtrType(Mod.getContext());
  FunctionAnalysisManager &FAM;
  // Cache for replacement functions.
  DenseMap<std::pair<LibFunc, uint64_t>, FunctionCallee> TokenAllocFunctions;
  // Selected mode.
  std::variant<IncrementMode, RandomMode, TypeHashMode,
               TypeHashPointerSplitMode>
      Mode;
};

bool AllocToken::instrumentFunction(Function &F) {
  // Do not apply any instrumentation for naked functions.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;
  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;
  // Don't touch available_externally functions, their actual body is elsewhere.
  if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage)
    return false;
  // Only instrument functions that have the sanitize_alloc_token attribute.
  if (!F.hasFnAttribute(Attribute::SanitizeAllocToken))
    return false;

  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
  SmallVector<std::pair<CallBase *, LibFunc>, 4> AllocCalls;

  // Collect all allocation calls to avoid iterator invalidation.
  for (Instruction &I : instructions(F)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB)
      continue;
    if (std::optional<LibFunc> Func = shouldInstrumentCall(*CB, TLI))
      AllocCalls.emplace_back(CB, Func.value());
  }

  bool Modified = false;
  for (auto &[CB, Func] : AllocCalls)
    Modified |= replaceAllocationCall(CB, Func, ORE, TLI);

  if (Modified)
    NumFunctionsInstrumented++;
  return Modified;
}

std::optional<LibFunc>
AllocToken::shouldInstrumentCall(const CallBase &CB,
                                 const TargetLibraryInfo &TLI) const {
  const Function *Callee = CB.getCalledFunction();
  if (!Callee)
    return std::nullopt;

  // Ignore nobuiltin of the CallBase, so that we can cover nobuiltin libcalls
  // if requested via isInstrumentableLibFunc(). Note that isAllocationFn() is
  // returning false for nobuiltin calls.
  LibFunc Func;
  if (TLI.getLibFunc(*Callee, Func)) {
    if (isInstrumentableLibFunc(Func, CB, TLI))
      return Func;
  } else if (Options.Extended && getAllocTokenMetadata(CB)) {
    return NotLibFunc;
  }

  return std::nullopt;
}

bool AllocToken::isInstrumentableLibFunc(LibFunc Func, const CallBase &CB,
                                         const TargetLibraryInfo &TLI) {
  if (ignoreInstrumentableLibFunc(Func))
    return false;

  if (isAllocationFn(&CB, &TLI))
    return true;

  switch (Func) {
  // These libfuncs don't return normal pointers, and are therefore not handled
  // by isAllocationFn().
  case LibFunc_posix_memalign:
  case LibFunc_size_returning_new:
  case LibFunc_size_returning_new_hot_cold:
  case LibFunc_size_returning_new_aligned:
  case LibFunc_size_returning_new_aligned_hot_cold:
    return true;

  // See comment above ClCoverReplaceableNew.
  case LibFunc_Znwj:
  case LibFunc_ZnwjRKSt9nothrow_t:
  case LibFunc_ZnwjSt11align_val_t:
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t:
  case LibFunc_Znwm:
  case LibFunc_Znwm12__hot_cold_t:
  case LibFunc_ZnwmRKSt9nothrow_t:
  case LibFunc_ZnwmRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_ZnwmSt11align_val_t:
  case LibFunc_ZnwmSt11align_val_t12__hot_cold_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_Znaj:
  case LibFunc_ZnajRKSt9nothrow_t:
  case LibFunc_ZnajSt11align_val_t:
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t:
  case LibFunc_Znam:
  case LibFunc_Znam12__hot_cold_t:
  case LibFunc_ZnamRKSt9nothrow_t:
  case LibFunc_ZnamRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_ZnamSt11align_val_t:
  case LibFunc_ZnamSt11align_val_t12__hot_cold_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
    return ClCoverReplaceableNew;

  default:
    return false;
  }
}

bool AllocToken::ignoreInstrumentableLibFunc(LibFunc Func) {
  switch (Func) {
  case LibFunc_strdup:
  case LibFunc_dunder_strdup:
  case LibFunc_strndup:
  case LibFunc_dunder_strndup:
    return true;
  default:
    return false;
  }
}

bool AllocToken::replaceAllocationCall(CallBase *CB, LibFunc Func,
                                       OptimizationRemarkEmitter &ORE,
                                       const TargetLibraryInfo &TLI) {
  uint64_t TokenID = getToken(*CB, ORE);

  FunctionCallee TokenAlloc = getTokenAllocFunction(*CB, TokenID, Func);
  if (!TokenAlloc)
    return false;
  NumAllocationsInstrumented++;

  if (Options.FastABI) {
    assert(TokenAlloc.getFunctionType()->getNumParams() == CB->arg_size());
    CB->setCalledFunction(TokenAlloc);
    return true;
  }

  IRBuilder<> IRB(CB);
  // Original args.
  SmallVector<Value *, 4> NewArgs{CB->args()};
  // Add token ID, truncated to IntPtrTy width.
  NewArgs.push_back(ConstantInt::get(IntPtrTy, TokenID));
  assert(TokenAlloc.getFunctionType()->getNumParams() == NewArgs.size());

  // Preserve invoke vs call semantics for exception handling.
  CallBase *NewCall;
  if (auto *II = dyn_cast<InvokeInst>(CB)) {
    NewCall = IRB.CreateInvoke(TokenAlloc, II->getNormalDest(),
                               II->getUnwindDest(), NewArgs);
  } else {
    NewCall = IRB.CreateCall(TokenAlloc, NewArgs);
    cast<CallInst>(NewCall)->setTailCall(CB->isTailCall());
  }
  NewCall->setCallingConv(CB->getCallingConv());
  NewCall->copyMetadata(*CB);
  NewCall->setAttributes(CB->getAttributes());

  // Replace all uses and delete the old call.
  CB->replaceAllUsesWith(NewCall);
  CB->eraseFromParent();
  return true;
}

FunctionCallee AllocToken::getTokenAllocFunction(const CallBase &CB,
                                                 uint64_t TokenID,
                                                 LibFunc OriginalFunc) {
  std::optional<std::pair<LibFunc, uint64_t>> Key;
  if (OriginalFunc != NotLibFunc) {
    Key = std::make_pair(OriginalFunc, Options.FastABI ? TokenID : 0);
    auto It = TokenAllocFunctions.find(*Key);
    if (It != TokenAllocFunctions.end())
      return It->second;
  }

  const Function *Callee = CB.getCalledFunction();
  if (!Callee)
    return FunctionCallee();
  const FunctionType *OldFTy = Callee->getFunctionType();
  if (OldFTy->isVarArg())
    return FunctionCallee();
  // Copy params, and append token ID type.
  Type *RetTy = OldFTy->getReturnType();
  SmallVector<Type *, 4> NewParams{OldFTy->params()};
  std::string TokenAllocName = ClFuncPrefix;
  if (Options.FastABI)
    TokenAllocName += utostr(TokenID) + "_";
  else
    NewParams.push_back(IntPtrTy); // token ID
  TokenAllocName += Callee->getName();
  FunctionType *NewFTy = FunctionType::get(RetTy, NewParams, false);
  FunctionCallee TokenAlloc = Mod.getOrInsertFunction(TokenAllocName, NewFTy);
  if (Function *F = dyn_cast<Function>(TokenAlloc.getCallee()))
    F->copyAttributesFrom(Callee); // preserve attrs

  if (Key.has_value())
    TokenAllocFunctions[*Key] = TokenAlloc;
  return TokenAlloc;
}

} // namespace

AllocTokenPass::AllocTokenPass(AllocTokenOptions Opts)
    : Options(std::move(Opts)) {}

PreservedAnalyses AllocTokenPass::run(Module &M, ModuleAnalysisManager &MAM) {
  AllocToken Pass(Options, M, MAM);
  bool Modified = false;

  for (Function &F : M) {
    if (F.empty())
      continue; // declaration
    Modified |= Pass.instrumentFunction(F);
  }

  return Modified ? PreservedAnalyses::none().preserveSet<CFGAnalyses>()
                  : PreservedAnalyses::all();
}
