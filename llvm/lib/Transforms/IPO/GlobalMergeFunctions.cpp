//===---- GlobalMergeFunctions.cpp - Global merge functions -------*- C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass defines the implementation of a function merging mechanism
// that utilizes a stable function hash to track differences in constants and
// create potential merge candidates. The process involves two rounds:
// 1. The first round collects stable function hashes and identifies merge
//    candidates with matching hashes. It also computes the set of parameters
//    that point to different constants during the stable function merge.
// 2. The second round leverages this collected global function information to
//    optimistically create a merged function in each module context, ensuring
//    correct transformation.
// Similar to the global outliner, this approach uses the linker's deduplication
// (ICF) to fold identical merged functions, thereby reducing the final binary
// size. The work is inspired by the concepts discussed in the following paper:
// https://dl.acm.org/doi/pdf/10.1145/3652032.3657575.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GlobalMergeFunctions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/CGData/CodeGenData.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "global-merge-func"

using namespace llvm;
using namespace llvm::support;

static cl::opt<bool>
    DisableGlobalMerging("disable-global-merging", cl::Hidden,
                         cl::desc("Disable global merging only by ignoring "
                                  "the codegen data generation or use. Local "
                                  "merging is still enabled within a module."),
                         cl::init(false));
static cl::opt<unsigned> GlobalMergingMinInstrs(
    "global-merging-min-instrs",
    cl::desc("The minimum instruction count required when merging functions."),
    cl::init(1), cl::Hidden);
static cl::opt<unsigned> GlobalMergingMaxParams(
    "global-merging-max-params",
    cl::desc(
        "The maximum number of parameters allowed when merging functions."),
    cl::init(std::numeric_limits<unsigned>::max()), cl::Hidden);
static cl::opt<unsigned> GlobalMergingParamOverhead(
    "global-merging-param-overhead",
    cl::desc("The overhead cost associated with each parameter when merging "
             "functions."),
    cl::init(2), cl::Hidden);
static cl::opt<unsigned>
    GlobalMergingCallOverhead("global-merging-call-overhead",
                              cl::desc("The overhead cost associated with each "
                                       "function call when merging functions."),
                              cl::init(1), cl::Hidden);
static cl::opt<unsigned> GlobalMergingExtraThreshold(
    "global-merging-extra-threshold",
    cl::desc("An additional cost threshold that must be exceeded for merging "
             "to be considered beneficial."),
    cl::init(0), cl::Hidden);

extern cl::opt<bool> EnableGlobalMergeFunc;

STATISTIC(NumMismatchedFunctionHashGlobalMergeFunction,
          "Number of mismatched function hash for global merge function");
STATISTIC(NumMismatchedInstCountGlobalMergeFunction,
          "Number of mismatched instruction count for global merge function");
STATISTIC(NumMismatchedConstHashGlobalMergeFunction,
          "Number of mismatched const hash for global merge function");
STATISTIC(NumMismatchedModuleIdGlobalMergeFunction,
          "Number of mismatched Module Id for global merge function");
STATISTIC(NumGlobalMergeFunctions,
          "Number of functions that are actually merged using function hash");
STATISTIC(NumAnalyzedModues, "Number of modules that are analyzed");
STATISTIC(NumAnalyzedFunctions, "Number of functions that are analyzed");
STATISTIC(NumEligibleFunctions, "Number of functions that are eligible");

/// Returns true if the \OpIdx operand of \p CI is the callee operand.
static bool isCalleeOperand(const CallBase *CI, unsigned OpIdx) {
  return &CI->getCalledOperandUse() == &CI->getOperandUse(OpIdx);
}

static bool canParameterizeCallOperand(const CallBase *CI, unsigned OpIdx) {
  if (CI->isInlineAsm())
    return false;
  Function *Callee = CI->getCalledOperand()
                         ? dyn_cast_or_null<Function>(
                               CI->getCalledOperand()->stripPointerCasts())
                         : nullptr;
  if (Callee) {
    if (Callee->isIntrinsic())
      return false;
    // objc_msgSend stubs must be called, and can't have their address taken.
    if (Callee->getName().starts_with("objc_msgSend$"))
      return false;
  }
  if (isCalleeOperand(CI, OpIdx) &&
      CI->getOperandBundle(LLVMContext::OB_ptrauth).has_value()) {
    // The operand is the callee and it has already been signed. Ignore this
    // because we cannot add another ptrauth bundle to the call instruction.
    return false;
  }
  return true;
}

bool isEligibleInstrunctionForConstantSharing(const Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::Call:
  case Instruction::Invoke:
    return true;
  default:
    return false;
  }
}

bool isEligibleOperandForConstantSharing(const Instruction *I, unsigned OpIdx) {
  assert(OpIdx < I->getNumOperands() && "Invalid operand index");

  if (!isEligibleInstrunctionForConstantSharing(I))
    return false;

  auto Opnd = I->getOperand(OpIdx);
  if (!isa<Constant>(Opnd))
    return false;

  if (const auto *CI = dyn_cast<CallBase>(I))
    return canParameterizeCallOperand(CI, OpIdx);

  return true;
}

/// Returns true if function \p F is eligible for merging.
bool isEligibleFunction(Function *F) {
  if (F->isDeclaration())
    return false;

  if (F->hasFnAttribute(llvm::Attribute::NoMerge))
    return false;

  if (F->hasAvailableExternallyLinkage())
    return false;

  if (F->getFunctionType()->isVarArg())
    return false;

  if (F->getCallingConv() == CallingConv::SwiftTail)
    return false;

  // If function contains callsites with musttail, if we merge
  // it, the merged function will have the musttail callsite, but
  // the number of parameters can change, thus the parameter count
  // of the callsite will mismatch with the function itself.
  for (const BasicBlock &BB : *F) {
    for (const Instruction &I : BB) {
      const auto *CB = dyn_cast<CallBase>(&I);
      if (CB && CB->isMustTailCall())
        return false;
    }
  }

  return true;
}

static bool
isEligibleInstrunctionForConstantSharingLocal(const Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::Call:
  case Instruction::Invoke:
    return true;
  default:
    return false;
  }
}

static bool ignoreOp(const Instruction *I, unsigned OpIdx) {
  assert(OpIdx < I->getNumOperands() && "Invalid operand index");

  if (!isEligibleInstrunctionForConstantSharingLocal(I))
    return false;

  if (!isa<Constant>(I->getOperand(OpIdx)))
    return false;

  if (const auto *CI = dyn_cast<CallBase>(I))
    return canParameterizeCallOperand(CI, OpIdx);

  return true;
}

static Value *createCast(IRBuilder<> &Builder, Value *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy->isStructTy()) {
    assert(DestTy->isStructTy());
    assert(SrcTy->getStructNumElements() == DestTy->getStructNumElements());
    Value *Result = PoisonValue::get(DestTy);
    for (unsigned int I = 0, E = SrcTy->getStructNumElements(); I < E; ++I) {
      Value *Element =
          createCast(Builder, Builder.CreateExtractValue(V, ArrayRef(I)),
                     DestTy->getStructElementType(I));

      Result = Builder.CreateInsertValue(Result, Element, ArrayRef(I));
    }
    return Result;
  }
  assert(!DestTy->isStructTy());
  if (auto *SrcAT = dyn_cast<ArrayType>(SrcTy)) {
    auto *DestAT = dyn_cast<ArrayType>(DestTy);
    assert(DestAT);
    assert(SrcAT->getNumElements() == DestAT->getNumElements());
    Value *Result = UndefValue::get(DestTy);
    for (unsigned int I = 0, E = SrcAT->getNumElements(); I < E; ++I) {
      Value *Element =
          createCast(Builder, Builder.CreateExtractValue(V, ArrayRef(I)),
                     DestAT->getElementType());

      Result = Builder.CreateInsertValue(Result, Element, ArrayRef(I));
    }
    return Result;
  }
  assert(!DestTy->isArrayTy());
  if (SrcTy->isIntegerTy() && DestTy->isPointerTy())
    return Builder.CreateIntToPtr(V, DestTy);
  else if (SrcTy->isPointerTy() && DestTy->isIntegerTy())
    return Builder.CreatePtrToInt(V, DestTy);
  else
    return Builder.CreateBitCast(V, DestTy);
}

void GlobalMergeFunc::analyze(Module &M) {
  ++NumAnalyzedModues;
  for (Function &Func : M) {
    ++NumAnalyzedFunctions;
    if (isEligibleFunction(&Func)) {
      ++NumEligibleFunctions;

      auto FI = llvm::StructuralHashWithDifferences(Func, ignoreOp);

      // Convert the operand map to a vector for a serialization-friendly
      // format.
      IndexOperandHashVecType IndexOperandHashes;
      for (auto &Pair : *FI.IndexOperandHashMap)
        IndexOperandHashes.emplace_back(Pair);

      StableFunction SF(FI.FunctionHash, get_stable_name(Func.getName()).str(),
                        M.getModuleIdentifier(), FI.IndexInstruction->size(),
                        std::move(IndexOperandHashes));

      LocalFunctionMap->insert(SF);
    }
  }
}

/// Tuple to hold function info to process merging.
struct FuncMergeInfo {
  StableFunctionMap::StableFunctionEntry *SF;
  Function *F;
  std::unique_ptr<IndexInstrMap> IndexInstruction;
};

// Given the func info, and the parameterized locations, create and return
// a new merged function by replacing the original constants with the new
// parameters.
static Function *createMergedFunction(FuncMergeInfo &FI,
                                      ArrayRef<Type *> ConstParamTypes,
                                      const ParamLocsVecTy &ParamLocsVec) {
  // Synthesize a new merged function name by appending ".Tgm" to the root
  // function's name.
  auto *MergedFunc = FI.F;
  auto NewFunctionName = MergedFunc->getName().str() + ".Tgm";
  auto *M = MergedFunc->getParent();
  assert(!M->getFunction(NewFunctionName));

  FunctionType *OrigTy = MergedFunc->getFunctionType();
  // Get the original params' types.
  SmallVector<Type *> ParamTypes(OrigTy->param_begin(), OrigTy->param_end());
  // Append const parameter types that are passed in.
  ParamTypes.append(ConstParamTypes.begin(), ConstParamTypes.end());
  FunctionType *FuncType =
      FunctionType::get(OrigTy->getReturnType(), ParamTypes, false);

  // Declare a new function
  Function *NewFunction =
      Function::Create(FuncType, MergedFunc->getLinkage(), NewFunctionName);
  if (auto *SP = MergedFunc->getSubprogram())
    NewFunction->setSubprogram(SP);
  NewFunction->copyAttributesFrom(MergedFunc);
  NewFunction->setDLLStorageClass(GlobalValue::DefaultStorageClass);

  NewFunction->setLinkage(GlobalValue::InternalLinkage);
  NewFunction->addFnAttr(Attribute::NoInline);

  // Add the new function before the root function.
  M->getFunctionList().insert(MergedFunc->getIterator(), NewFunction);

  // Move the body of MergedFunc into the NewFunction.
  NewFunction->splice(NewFunction->begin(), MergedFunc);

  // Update the original args by the new args.
  auto NewArgIter = NewFunction->arg_begin();
  for (Argument &OrigArg : MergedFunc->args()) {
    Argument &NewArg = *NewArgIter++;
    OrigArg.replaceAllUsesWith(&NewArg);
  }

  // Replace the original Constants by the new args.
  unsigned NumOrigArgs = MergedFunc->arg_size();
  for (unsigned ParamIdx = 0; ParamIdx < ParamLocsVec.size(); ++ParamIdx) {
    Argument *NewArg = NewFunction->getArg(NumOrigArgs + ParamIdx);
    for (auto [InstIndex, OpndIndex] : ParamLocsVec[ParamIdx]) {
      auto *Inst = FI.IndexInstruction->lookup(InstIndex);
      auto *OrigC = Inst->getOperand(OpndIndex);
      if (OrigC->getType() != NewArg->getType()) {
        IRBuilder<> Builder(Inst->getParent(), Inst->getIterator());
        Inst->setOperand(OpndIndex,
                         createCast(Builder, NewArg, OrigC->getType()));
      } else
        Inst->setOperand(OpndIndex, NewArg);
    }
  }

  return NewFunction;
}

// Given the original function (Thunk) and the merged function (ToFunc), create
// a thunk to the merged function.

static void createThunk(FuncMergeInfo &FI, ArrayRef<Constant *> Params,
                        Function *ToFunc) {
  auto *Thunk = FI.F;

  assert(Thunk->arg_size() + Params.size() ==
         ToFunc->getFunctionType()->getNumParams());
  Thunk->dropAllReferences();

  BasicBlock *BB = BasicBlock::Create(Thunk->getContext(), "", Thunk);
  IRBuilder<> Builder(BB);

  SmallVector<Value *> Args;
  unsigned ParamIdx = 0;
  FunctionType *ToFuncTy = ToFunc->getFunctionType();

  // Add arguments which are passed through Thunk.
  for (Argument &AI : Thunk->args()) {
    Args.push_back(createCast(Builder, &AI, ToFuncTy->getParamType(ParamIdx)));
    ++ParamIdx;
  }

  // Add new arguments defined by Params.
  for (auto *Param : Params) {
    assert(ParamIdx < ToFuncTy->getNumParams());
    // FIXME: do not support signing
    Args.push_back(
        createCast(Builder, Param, ToFuncTy->getParamType(ParamIdx)));
    ++ParamIdx;
  }

  CallInst *CI = Builder.CreateCall(ToFunc, Args);
  bool isSwiftTailCall = ToFunc->getCallingConv() == CallingConv::SwiftTail &&
                         Thunk->getCallingConv() == CallingConv::SwiftTail;
  CI->setTailCallKind(isSwiftTailCall ? llvm::CallInst::TCK_MustTail
                                      : llvm::CallInst::TCK_Tail);
  CI->setCallingConv(ToFunc->getCallingConv());
  CI->setAttributes(ToFunc->getAttributes());
  if (Thunk->getReturnType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(createCast(Builder, CI, Thunk->getReturnType()));
}

// Check if the old merged/optimized IndexOperandHashMap is compatible with
// the current IndexOperandHashMap. An operand hash may not be stable across
// different builds due to varying modules combined. To address this, we relax
// the hash check condition by comparing Const hash patterns instead of absolute
// hash values. For example, let's assume we have three Consts located at idx1,
// idx3, and idx6, where their corresponding hashes are hash1, hash2, and hash1
// in the old merged map below:
//   Old (Merged): [(idx1, hash1), (idx3, hash2), (idx6, hash1)]
//   Current: [(idx1, hash1'), (idx3, hash2'), (idx6, hash1')]
// If the current function also has three Consts in the same locations,
// with hash sequences hash1', hash2', and hash1' where the first and third
// are the same as the old hash sequences, we consider them matched.
static bool checkConstHashCompatible(
    const DenseMap<IndexPair, stable_hash> &OldInstOpndIndexToConstHash,
    const DenseMap<IndexPair, stable_hash> &CurrInstOpndIndexToConstHash) {

  DenseMap<stable_hash, stable_hash> OldHashToCurrHash;
  for (const auto &[Index, OldHash] : OldInstOpndIndexToConstHash) {
    auto It = CurrInstOpndIndexToConstHash.find(Index);
    if (It == CurrInstOpndIndexToConstHash.end())
      return false;

    auto CurrHash = It->second;
    auto J = OldHashToCurrHash.find(OldHash);
    if (J == OldHashToCurrHash.end())
      OldHashToCurrHash.insert({OldHash, CurrHash});
    else if (J->second != CurrHash)
      return false;
  }

  return true;
}

// Validate the locations pointed by a param has the same hash and Constant.
static bool
checkConstLocationCompatible(const StableFunctionMap::StableFunctionEntry &SF,
                             const IndexInstrMap &IndexInstruction,
                             const ParamLocsVecTy &ParamLocsVec) {
  for (auto &ParamLocs : ParamLocsVec) {
    std::optional<stable_hash> OldHash;
    std::optional<Constant *> OldConst;
    for (auto &Loc : ParamLocs) {
      assert(SF.IndexOperandHashMap->count(Loc));
      auto CurrHash = SF.IndexOperandHashMap.get()->at(Loc);
      auto [InstIndex, OpndIndex] = Loc;
      assert(InstIndex < IndexInstruction.size());
      const auto *Inst = IndexInstruction.lookup(InstIndex);
      auto *CurrConst = cast<Constant>(Inst->getOperand(OpndIndex));
      if (!OldHash) {
        OldHash = CurrHash;
        OldConst = CurrConst;
      } else if (CurrConst != *OldConst || CurrHash != *OldHash)
        return false;
    }
  }
  return true;
}

static ParamLocsVecTy computeParamInfo(
    const SmallVector<std::unique_ptr<StableFunctionMap::StableFunctionEntry>>
        &SFS) {
  std::map<std::vector<stable_hash>, ParamLocs> HashSeqToLocs;
  auto &RSF = *SFS[0];
  unsigned StableFunctionCount = SFS.size();

  for (auto &[IndexPair, Hash] : *RSF.IndexOperandHashMap) {
    // Const hash sequence across stable functions.
    // We will allocate a parameter per unique hash squence.
    // can't use SmallVector as key
    std::vector<stable_hash> ConstHashSeq;
    ConstHashSeq.push_back(Hash);
    bool Identical = true;
    for (unsigned J = 1; J < StableFunctionCount; ++J) {
      auto &SF = SFS[J];
      assert(SF->IndexOperandHashMap->count(IndexPair));
      auto SHash = (*SF->IndexOperandHashMap)[IndexPair];
      if (Hash != SHash)
        Identical = false;
      ConstHashSeq.push_back(SHash);
    }

    if (Identical)
      continue;

    // For each unique Const hash sequence (parameter), add the locations.
    HashSeqToLocs[ConstHashSeq].push_back(IndexPair);
  }

  ParamLocsVecTy ParamLocsVec;
  for (auto &[HashSeq, Locs] : HashSeqToLocs) {
    ParamLocsVec.push_back(std::move(Locs));
    std::sort(
        ParamLocsVec.begin(), ParamLocsVec.end(),
        [&](const ParamLocs &L, const ParamLocs &R) { return L[0] < R[0]; });
  }
  return ParamLocsVec;
}

static bool isProfitable(
    const SmallVector<std::unique_ptr<StableFunctionMap::StableFunctionEntry>>
        &SFS,
    const Function *F) {
  // No interest if the number of candidates are less than 2.
  unsigned StableFunctionCount = SFS.size();
  if (StableFunctionCount < 2)
    return false;

  unsigned InstCount = SFS[0]->InstCount;
  if (InstCount < GlobalMergingMinInstrs)
    return false;

  unsigned ParamCount = SFS[0]->IndexOperandHashMap->size();
  unsigned TotalParamCount = ParamCount + F->getFunctionType()->getNumParams();
  if (TotalParamCount > GlobalMergingMaxParams)
    return false;

  unsigned Benefit = InstCount * (StableFunctionCount - 1);
  unsigned Cost =
      (GlobalMergingParamOverhead * ParamCount + GlobalMergingCallOverhead) *
          StableFunctionCount +
      GlobalMergingExtraThreshold;

  bool Result = Benefit > Cost;
  LLVM_DEBUG(dbgs() << "isProfitable: Function = " << F->getName() << ", "
                    << "StableFunctionCount = " << StableFunctionCount
                    << ", InstCount = " << InstCount
                    << ", ParamCount = " << ParamCount
                    << ", Benefit = " << Benefit << ", Cost = " << Cost
                    << ", Result = " << (Result ? "true" : "false") << "\n");
  return Result;
}

bool GlobalMergeFunc::merge(Module &M, const StableFunctionMap *FunctionMap) {
  bool Changed = false;

  // Build a map from stable function name to function.
  StringMap<Function *> StableNameToFuncMap;
  for (auto &F : M)
    StableNameToFuncMap[get_stable_name(F.getName())] = &F;
  // Track merged functions
  DenseSet<Function *> MergedFunctions;

  auto ModId = M.getModuleIdentifier();
  for (auto &[Hash, SFS] : FunctionMap->getFunctionMap()) {
    // Compute the parameter locations based on the unique hash sequences
    // across the candidates.
    auto ParamLocsVec = computeParamInfo(SFS);
    LLVM_DEBUG({
      dbgs() << "[GlobalMergeFunc] Merging hash: " << Hash << " with Params "
             << ParamLocsVec.size() << "\n";
    });

    Function *MergedFunc = nullptr;
    std::string MergedModId;
    SmallVector<FuncMergeInfo> FuncMergeInfos;
    for (auto &SF : SFS) {
      // Get the function from the stable name.
      auto I = StableNameToFuncMap.find(
          *FunctionMap->getNameForId(SF->FunctionNameId));
      if (I == StableNameToFuncMap.end())
        continue;
      Function *F = I->second;
      assert(F);
      // Skip if the function has been merged before.
      if (MergedFunctions.count(F))
        continue;
      // Consider the function if it is eligible for merging.
      if (!isEligibleFunction(F))
        continue;

      auto FI = llvm::StructuralHashWithDifferences(*F, ignoreOp);
      uint64_t FuncHash = FI.FunctionHash;
      if (Hash != FuncHash) {
        ++NumMismatchedFunctionHashGlobalMergeFunction;
        continue;
      }

      if (SF->InstCount != FI.IndexInstruction->size()) {
        ++NumMismatchedInstCountGlobalMergeFunction;
        continue;
      }
      bool HasValidSharedConst = true;
      for (auto &[Index, Hash] : *SF->IndexOperandHashMap) {
        auto [InstIndex, OpndIndex] = Index;
        assert(InstIndex < FI.IndexInstruction->size());
        auto *Inst = FI.IndexInstruction->lookup(InstIndex);
        if (!isEligibleOperandForConstantSharing(Inst, OpndIndex)) {
          HasValidSharedConst = false;
          break;
        }
      }
      if (!HasValidSharedConst) {
        ++NumMismatchedConstHashGlobalMergeFunction;
        continue;
      }
      if (!checkConstHashCompatible(*SF->IndexOperandHashMap,
                                    *FI.IndexOperandHashMap)) {
        ++NumMismatchedConstHashGlobalMergeFunction;
        continue;
      }
      if (!checkConstLocationCompatible(*SF, *FI.IndexInstruction,
                                        ParamLocsVec)) {
        ++NumMismatchedConstHashGlobalMergeFunction;
        continue;
      }

      if (!isProfitable(SFS, F))
        break;

      if (MergedFunc) {
        // Check if the matched functions fall into the same (first) module.
        // This module check is not strictly necessary as the functions can move
        // around. We just want to avoid merging functions from different
        // modules than the first one in the functon map, as they may not end up
        // with not being ICFed by the linker.
        if (MergedModId != *FunctionMap->getNameForId(SF->ModuleNameId)) {
          ++NumMismatchedModuleIdGlobalMergeFunction;
          continue;
        }
      } else {
        MergedFunc = F;
        MergedModId = *FunctionMap->getNameForId(SF->ModuleNameId);
      }

      FuncMergeInfos.push_back({SF.get(), F, std::move(FI.IndexInstruction)});
      MergedFunctions.insert(F);
    }
    unsigned FuncMergeInfoSize = FuncMergeInfos.size();
    if (FuncMergeInfoSize == 0)
      continue;

    LLVM_DEBUG({
      dbgs() << "[GlobalMergeFunc] Merging function count " << FuncMergeInfoSize
             << " in  " << ModId << "\n";
    });

    for (auto &FMI : FuncMergeInfos) {
      Changed = true;

      // We've already validated all locations of constant operands pointed by
      // the parameters. Populate parameters pointing to the original constants.
      SmallVector<Constant *> Params;
      SmallVector<Type *> ParamTypes;
      for (auto &ParamLocs : ParamLocsVec) {
        assert(!ParamLocs.empty());
        auto &[InstIndex, OpndIndex] = ParamLocs[0];
        auto *Inst = FMI.IndexInstruction->lookup(InstIndex);
        auto *Opnd = cast<Constant>(Inst->getOperand(OpndIndex));
        Params.push_back(Opnd);
        ParamTypes.push_back(Opnd->getType());
      }

      // Create a merged function derived from the current function.
      Function *MergedFunc =
          createMergedFunction(FMI, ParamTypes, ParamLocsVec);

      LLVM_DEBUG({
        dbgs() << "[GlobalMergeFunc] Merged function (hash:" << FMI.SF->Hash
               << ") " << MergedFunc->getName() << " generated from "
               << FMI.F->getName() << ":\n";
        MergedFunc->dump();
      });

      // Transform the current function into a thunk that calls the merged
      // function.
      createThunk(FMI, Params, MergedFunc);
      LLVM_DEBUG({
        dbgs() << "[GlobalMergeFunc] Thunk generated: \n";
        FMI.F->dump();
      });
      ++NumGlobalMergeFunctions;
    }
  }

  return Changed;
}

char GlobalMergeFunc::ID = 0;
INITIALIZE_PASS_BEGIN(GlobalMergeFunc, "global-merge-func",
                      "Global merge function pass", false, false)
INITIALIZE_PASS_END(GlobalMergeFunc, "global-merge-func",
                    "Global merge function pass", false, false)

StringRef GlobalMergeFunc::getPassName() const {
  return "Global Merge Functions";
}

void GlobalMergeFunc::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addUsedIfAvailable<ImmutableModuleSummaryIndexWrapperPass>();
  AU.setPreservesAll();
  ModulePass::getAnalysisUsage(AU);
}

GlobalMergeFunc::GlobalMergeFunc() : ModulePass(ID) {
  initializeGlobalMergeFuncPass(*llvm::PassRegistry::getPassRegistry());
}

namespace llvm {
Pass *createGlobalMergeFuncPass() { return new GlobalMergeFunc(); }
} // namespace llvm

void GlobalMergeFunc::initializeMergerMode(const Module &M) {
  LocalFunctionMap = std::make_unique<StableFunctionMap>();

  if (DisableGlobalMerging)
    return;

  if (auto *IndexWrapperPass =
          getAnalysisIfAvailable<ImmutableModuleSummaryIndexWrapperPass>()) {
    auto *TheIndex = IndexWrapperPass->getIndex();
    // (Full)LTO module does not have functions added to the index.
    // In this case, we run a local merger without using codegen data.
    if (TheIndex && !TheIndex->hasExportedFunctions(M))
      return;
  }

  if (cgdata::emitCGData())
    MergerMode = HashFunctionMode::BuildingHashFuncion;
  else if (cgdata::hasStableFunctionMap())
    MergerMode = HashFunctionMode::UsingHashFunction;
}

void GlobalMergeFunc::emitFunctionMap(Module &M) {
  LLVM_DEBUG({
    dbgs() << "Emit function map. Size: " << LocalFunctionMap->size() << "\n";
  });
  // No need to emit the function map if it is empty.
  if (LocalFunctionMap->empty())
    return;
  SmallVector<char> Buf;
  raw_svector_ostream OS(Buf);

  StableFunctionMapRecord::serialize(OS, LocalFunctionMap.get());

  std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer(
      OS.str(), "in-memory stable function map", false);

  Triple TT(M.getTargetTriple());
  embedBufferInModule(M, *Buffer.get(),
                      getCodeGenDataSectionName(CG_merge, TT.getObjectFormat()),
                      Align(4));
}

bool GlobalMergeFunc::runOnModule(Module &M) {
  initializeMergerMode(M);

  const StableFunctionMap *FuncMap;
  if (MergerMode == HashFunctionMode::UsingHashFunction) {
    // Use the prior CG data to optimistically create global merge candidates.
    FuncMap = cgdata::getStableFunctionMap();
  } else {
    analyze(M);
    // Emit the local function map to the custom section, __llvm_merge before
    // finalizing it.
    if (MergerMode == HashFunctionMode::BuildingHashFuncion)
      emitFunctionMap(M);
    LocalFunctionMap->finalize();
    FuncMap = LocalFunctionMap.get();
  }

  return merge(M, FuncMap);
}
