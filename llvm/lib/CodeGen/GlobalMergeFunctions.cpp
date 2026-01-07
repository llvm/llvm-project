//===---- GlobalMergeFunctions.cpp - Global merge functions -------*- C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the global merge function pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalMergeFunctions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/CGData/CodeGenData.h"
#include "llvm/CGData/CodeGenDataWriter.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "global-merge-func"

using namespace llvm;
using namespace llvm::support;

static cl::opt<bool> DisableCGDataForMerging(
    "disable-cgdata-for-merging", cl::Hidden,
    cl::desc("Disable codegen data for function merging. Local "
             "merging is still enabled within a module."),
    cl::init(false));

STATISTIC(NumMergedFunctions,
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
    auto Name = Callee->getName();
    // objc_msgSend stubs must be called, and can't have their address taken.
    if (Name.starts_with("objc_msgSend$"))
      return false;
    // Calls to dtrace probes must generate unique patchpoints.
    if (Name.starts_with("__dtrace"))
      return false;
  }
  if (isCalleeOperand(CI, OpIdx)) {
    // The operand is the callee and it has already been signed. Ignore this
    // because we cannot add another ptrauth bundle to the call instruction.
    if (CI->getOperandBundle(LLVMContext::OB_ptrauth).has_value())
      return false;
  } else {
    // The target of the arc-attached call must be a constant and cannot be
    // parameterized.
    if (CI->isOperandBundleOfType(LLVMContext::OB_clang_arc_attachedcall,
                                  OpIdx))
      return false;
  }
  return true;
}

/// Returns true if function \p F is eligible for merging.
bool isEligibleFunction(Function *F) {
  if (F->isDeclaration())
    return false;

  if (F->hasFnAttribute(llvm::Attribute::NoMerge) ||
      F->hasFnAttribute(llvm::Attribute::AlwaysInline))
    return false;

  if (F->hasAvailableExternallyLinkage())
    return false;

  if (F->getFunctionType()->isVarArg())
    return false;

  if (F->getCallingConv() == CallingConv::SwiftTail)
    return false;

  // Unnamed functions are skipped for simplicity.
  if (!F->hasName())
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

static bool isEligibleInstructionForConstantSharing(const Instruction *I) {
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

// This function takes an instruction, \p I, and an operand index, \p OpIdx.
// It returns true if the operand should be ignored in the hash computation.
// If \p OpIdx is out of range based on the other instruction context, it cannot
// be ignored.
static bool ignoreOp(const Instruction *I, unsigned OpIdx) {
  if (OpIdx >= I->getNumOperands())
    return false;

  if (!isEligibleInstructionForConstantSharing(I))
    return false;

  if (!isa<Constant>(I->getOperand(OpIdx)))
    return false;

  if (const auto *CI = dyn_cast<CallBase>(I))
    return canParameterizeCallOperand(CI, OpIdx);

  return true;
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
  IndexInstrMap *IndexInstruction;
  FuncMergeInfo(StableFunctionMap::StableFunctionEntry *SF, Function *F,
                IndexInstrMap *IndexInstruction)
      : SF(SF), F(F), IndexInstruction(std::move(IndexInstruction)) {}
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
  std::string NewFunctionName =
      MergedFunc->getName().str() + GlobalMergeFunc::MergingInstanceSuffix;
  auto *M = MergedFunc->getParent();
  assert(!M->getFunction(NewFunctionName));

  FunctionType *OrigTy = MergedFunc->getFunctionType();
  // Get the original params' types.
  SmallVector<Type *> ParamTypes(OrigTy->param_begin(), OrigTy->param_end());
  // Append const parameter types that are passed in.
  ParamTypes.append(ConstParamTypes.begin(), ConstParamTypes.end());
  FunctionType *FuncType = FunctionType::get(OrigTy->getReturnType(),
                                             ParamTypes, /*isVarArg=*/false);

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
                         Builder.CreateAggregateCast(NewArg, OrigC->getType()));
      } else {
        Inst->setOperand(OpndIndex, NewArg);
      }
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
    Args.push_back(
        Builder.CreateAggregateCast(&AI, ToFuncTy->getParamType(ParamIdx)));
    ++ParamIdx;
  }

  // Add new arguments defined by Params.
  for (auto *Param : Params) {
    assert(ParamIdx < ToFuncTy->getNumParams());
    Args.push_back(
        Builder.CreateAggregateCast(Param, ToFuncTy->getParamType(ParamIdx)));
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
    Builder.CreateRet(Builder.CreateAggregateCast(CI, Thunk->getReturnType()));
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
      auto CurrHash = SF.IndexOperandHashMap->at(Loc);
      auto [InstIndex, OpndIndex] = Loc;
      assert(InstIndex < IndexInstruction.size());
      const auto *Inst = IndexInstruction.lookup(InstIndex);
      auto *CurrConst = cast<Constant>(Inst->getOperand(OpndIndex));
      if (!OldHash) {
        OldHash = CurrHash;
        OldConst = CurrConst;
      } else if (CurrConst != *OldConst || CurrHash != *OldHash) {
        return false;
      }
    }
  }
  return true;
}

static ParamLocsVecTy
computeParamInfo(const StableFunctionMap::StableFunctionEntries &SFS) {
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
      auto SHash = SF->IndexOperandHashMap->at(IndexPair);
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
  for (auto &[HashSeq, Locs] : HashSeqToLocs)
    ParamLocsVec.push_back(std::move(Locs));

  llvm::sort(ParamLocsVec, [&](const ParamLocs &L, const ParamLocs &R) {
    return L[0] < R[0];
  });

  return ParamLocsVec;
}

bool GlobalMergeFunc::merge(Module &M, const StableFunctionMap *FunctionMap) {
  bool Changed = false;

  // Collect stable functions related to the current module.
  DenseMap<stable_hash, SmallVector<std::pair<Function *, FunctionHashInfo>>>
      HashToFuncs;
  for (auto &F : M) {
    if (!isEligibleFunction(&F))
      continue;
    auto FI = llvm::StructuralHashWithDifferences(F, ignoreOp);
    if (FunctionMap->contains(FI.FunctionHash))
      HashToFuncs[FI.FunctionHash].emplace_back(&F, std::move(FI));
  }

  for (auto &[Hash, Funcs] : HashToFuncs) {
    std::optional<ParamLocsVecTy> ParamLocsVec;
    SmallVector<FuncMergeInfo> FuncMergeInfos;
    auto &SFS = FunctionMap->at(Hash);
    assert(!SFS.empty());
    auto &RFS = SFS[0];

    // Iterate functions with the same hash.
    for (auto &[F, FI] : Funcs) {
      // Check if the function is compatible with any stable function
      // in terms of the number of instructions and ignored operands.
      if (RFS->InstCount != FI.IndexInstruction->size())
        continue;

      auto hasValidSharedConst = [&](StableFunctionMap::StableFunctionEntry *SF,
                                     FunctionHashInfo &FHI) {
        for (auto &[Index, Hash] : *SF->IndexOperandHashMap) {
          auto [InstIndex, OpndIndex] = Index;
          assert(InstIndex < FHI.IndexInstruction->size());
          auto *Inst = FHI.IndexInstruction->lookup(InstIndex);
          if (!ignoreOp(Inst, OpndIndex))
            return false;
        }
        return true;
      };
      if (!hasValidSharedConst(RFS.get(), FI))
        continue;

      for (auto &SF : SFS) {
        assert(SF->InstCount == FI.IndexInstruction->size());
        assert(hasValidSharedConst(SF.get(), FI));
        // Check if there is any stable function that is compatiable with the
        // current one.
        if (!checkConstHashCompatible(*SF->IndexOperandHashMap,
                                      *FI.IndexOperandHashMap))
          continue;
        if (!ParamLocsVec.has_value()) {
          ParamLocsVec = computeParamInfo(SFS);
          LLVM_DEBUG(dbgs() << "[GlobalMergeFunc] Merging hash: " << Hash
                            << " with Params " << ParamLocsVec->size() << "\n");
        }
        if (!checkConstLocationCompatible(*SF, *FI.IndexInstruction,
                                          *ParamLocsVec))
          continue;

        // If a stable function matching the current one is found,
        // create a candidate for merging and proceed to the next function.
        FuncMergeInfos.emplace_back(SF.get(), F, FI.IndexInstruction.get());
        break;
      }
    }
    unsigned FuncMergeInfoSize = FuncMergeInfos.size();
    if (FuncMergeInfoSize == 0)
      continue;

    LLVM_DEBUG(dbgs() << "[GlobalMergeFunc] Merging function count "
                      << FuncMergeInfoSize << " for hash:  " << Hash << "\n");

    for (auto &FMI : FuncMergeInfos) {
      Changed = true;

      // We've already validated all locations of constant operands pointed by
      // the parameters. Populate parameters pointing to the original constants.
      SmallVector<Constant *> Params;
      SmallVector<Type *> ParamTypes;
      for (auto &ParamLocs : *ParamLocsVec) {
        assert(!ParamLocs.empty());
        auto &[InstIndex, OpndIndex] = ParamLocs[0];
        auto *Inst = FMI.IndexInstruction->lookup(InstIndex);
        auto *Opnd = cast<Constant>(Inst->getOperand(OpndIndex));
        Params.push_back(Opnd);
        ParamTypes.push_back(Opnd->getType());
      }

      // Create a merged function derived from the current function.
      Function *MergedFunc =
          createMergedFunction(FMI, ParamTypes, *ParamLocsVec);

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
      ++NumMergedFunctions;
    }
  }

  return Changed;
}

void GlobalMergeFunc::initializeMergerMode(const Module &M) {
  // Initialize the local function map regardless of the merger mode.
  LocalFunctionMap = std::make_unique<StableFunctionMap>();

  // Disable codegen data for merging. The local merge is still enabled.
  if (DisableCGDataForMerging)
    return;

  // (Full)LTO module does not have functions added to the index.
  // In this case, we run a local merger without using codegen data.
  if (Index && !Index->hasExportedFunctions(M))
    return;

  if (cgdata::emitCGData())
    MergerMode = HashFunctionMode::BuildingHashFuncion;
  else if (cgdata::hasStableFunctionMap())
    MergerMode = HashFunctionMode::UsingHashFunction;
}

void GlobalMergeFunc::emitFunctionMap(Module &M) {
  LLVM_DEBUG(dbgs() << "Emit function map. Size: " << LocalFunctionMap->size()
                    << "\n");
  // No need to emit the function map if it is empty.
  if (LocalFunctionMap->empty())
    return;
  SmallVector<char> Buf;
  raw_svector_ostream OS(Buf);

  std::vector<CGDataPatchItem> PatchItems;
  StableFunctionMapRecord::serialize(OS, LocalFunctionMap.get(), PatchItems);
  CGDataOStream COS(OS);
  COS.patch(PatchItems);

  std::unique_ptr<MemoryBuffer> Buffer = MemoryBuffer::getMemBuffer(
      OS.str(), "in-memory stable function map", false);

  Triple TT(M.getTargetTriple());
  embedBufferInModule(M, *Buffer,
                      getCodeGenDataSectionName(CG_merge, TT.getObjectFormat()),
                      Align(4));
}

bool GlobalMergeFunc::run(Module &M) {
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

namespace {

class GlobalMergeFuncPassWrapper : public ModulePass {

public:
  static char ID;

  GlobalMergeFuncPassWrapper();

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addUsedIfAvailable<ImmutableModuleSummaryIndexWrapperPass>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Global Merge Functions"; }

  bool runOnModule(Module &M) override;
};

} // namespace

char GlobalMergeFuncPassWrapper::ID = 0;
INITIALIZE_PASS(GlobalMergeFuncPassWrapper, "global-merge-func",
                "Global merge function pass", false, false)

ModulePass *llvm::createGlobalMergeFuncPass() {
  return new GlobalMergeFuncPassWrapper();
}

GlobalMergeFuncPassWrapper::GlobalMergeFuncPassWrapper() : ModulePass(ID) {
  initializeGlobalMergeFuncPassWrapperPass(
      *llvm::PassRegistry::getPassRegistry());
}

bool GlobalMergeFuncPassWrapper::runOnModule(Module &M) {
  const ModuleSummaryIndex *Index = nullptr;
  if (auto *IndexWrapperPass =
          getAnalysisIfAvailable<ImmutableModuleSummaryIndexWrapperPass>())
    Index = IndexWrapperPass->getIndex();

  return GlobalMergeFunc(Index).run(M);
}

PreservedAnalyses GlobalMergeFuncPass::run(Module &M,
                                           AnalysisManager<Module> &AM) {
  bool Changed = GlobalMergeFunc(ImportSummary).run(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
