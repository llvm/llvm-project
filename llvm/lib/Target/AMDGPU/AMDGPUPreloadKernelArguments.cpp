//===- AMDGPUPreloadKernelArguments.cpp - Preload Kernel Arguments --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass preloads kernel arguments into user_data SGPRs before kernel
/// execution begins. The number of registers available for preloading depends
/// on the number of free user SGPRs, up to the hardware's maximum limit.
/// Implicit arguments enabled in the kernel descriptor are allocated first,
/// followed by SGPRs used for preloaded kernel arguments. (Reference:
/// https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state)
/// Additionally, hidden kernel arguments may be preloaded, in which case they
/// are appended to the kernel signature after explicit arguments. Preloaded
/// arguments will be marked with `inreg`.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "amdgpu-preload-kernel-arguments"

using namespace llvm;

static cl::opt<unsigned> KernargPreloadCount(
    "amdgpu-kernarg-preload-count",
    cl::desc("How many kernel arguments to preload onto SGPRs"), cl::init(0));

static cl::opt<bool>
    EnableKernargPreload("amdgpu-kernarg-preload",
                         cl::desc("Enable preload kernel arguments to SGPRs"),
                         cl::init(true));

namespace {

class AMDGPUPreloadKernelArgumentsLegacy : public ModulePass {
  const GCNTargetMachine *TM;

public:
  static char ID;
  explicit AMDGPUPreloadKernelArgumentsLegacy(
      const GCNTargetMachine *TM = nullptr);

  StringRef getPassName() const override {
    return "AMDGPU Preload Kernel Arguments";
  }

  bool runOnModule(Module &M) override;
};

class PreloadKernelArgInfo {
private:
  Function &F;
  const GCNSubtarget &ST;
  unsigned NumFreeUserSGPRs;

  enum HiddenArg : unsigned {
    HIDDEN_BLOCK_COUNT_X,
    HIDDEN_BLOCK_COUNT_Y,
    HIDDEN_BLOCK_COUNT_Z,
    HIDDEN_GROUP_SIZE_X,
    HIDDEN_GROUP_SIZE_Y,
    HIDDEN_GROUP_SIZE_Z,
    HIDDEN_REMAINDER_X,
    HIDDEN_REMAINDER_Y,
    HIDDEN_REMAINDER_Z,
    END_HIDDEN_ARGS
  };

  // Stores information about a specific hidden argument.
  struct HiddenArgInfo {
    // Offset in bytes from the location in the kernearg segment pointed to by
    // the implicitarg pointer.
    uint8_t Offset;
    // The size of the hidden argument in bytes.
    uint8_t Size;
    // The name of the hidden argument in the kernel signature.
    const char *Name;
  };

  static constexpr HiddenArgInfo HiddenArgs[END_HIDDEN_ARGS] = {
      {0, 4, "_hidden_block_count_x"}, {4, 4, "_hidden_block_count_y"},
      {8, 4, "_hidden_block_count_z"}, {12, 2, "_hidden_group_size_x"},
      {14, 2, "_hidden_group_size_y"}, {16, 2, "_hidden_group_size_z"},
      {18, 2, "_hidden_remainder_x"},  {20, 2, "_hidden_remainder_y"},
      {22, 2, "_hidden_remainder_z"}};

  static HiddenArg getHiddenArgFromOffset(unsigned Offset) {
    for (unsigned I = 0; I < END_HIDDEN_ARGS; ++I)
      if (HiddenArgs[I].Offset == Offset)
        return static_cast<HiddenArg>(I);

    return END_HIDDEN_ARGS;
  }

  static Type *getHiddenArgType(LLVMContext &Ctx, HiddenArg HA) {
    if (HA < END_HIDDEN_ARGS)
      return Type::getIntNTy(Ctx, HiddenArgs[HA].Size * 8);

    llvm_unreachable("Unexpected hidden argument.");
  }

  static const char *getHiddenArgName(HiddenArg HA) {
    if (HA < END_HIDDEN_ARGS)
      return HiddenArgs[HA].Name;

    llvm_unreachable("Unexpected hidden argument.");
  }

  // Clones the function after adding implicit arguments to the argument list
  // and returns the new updated function. Preloaded implicit arguments are
  // added up to and including the last one that will be preloaded, indicated by
  // LastPreloadIndex. Currently preloading is only performed on the totality of
  // sequential data from the kernarg segment including implicit (hidden)
  // arguments. This means that all arguments up to the last preloaded argument
  // will also be preloaded even if that data is unused.
  Function *cloneFunctionWithPreloadImplicitArgs(unsigned LastPreloadIndex) {
    FunctionType *FT = F.getFunctionType();
    LLVMContext &Ctx = F.getParent()->getContext();
    SmallVector<Type *, 16> FTypes(FT->param_begin(), FT->param_end());
    for (unsigned I = 0; I <= LastPreloadIndex; ++I)
      FTypes.push_back(getHiddenArgType(Ctx, HiddenArg(I)));

    FunctionType *NFT =
        FunctionType::get(FT->getReturnType(), FTypes, FT->isVarArg());
    Function *NF =
        Function::Create(NFT, F.getLinkage(), F.getAddressSpace(), F.getName());

    NF->copyAttributesFrom(&F);
    NF->copyMetadata(&F, 0);

    F.getParent()->getFunctionList().insert(F.getIterator(), NF);
    NF->takeName(&F);
    NF->splice(NF->begin(), &F);

    Function::arg_iterator NFArg = NF->arg_begin();
    for (Argument &Arg : F.args()) {
      Arg.replaceAllUsesWith(&*NFArg);
      NFArg->takeName(&Arg);
      ++NFArg;
    }

    AttrBuilder AB(Ctx);
    AB.addAttribute(Attribute::InReg);
    AB.addAttribute("amdgpu-hidden-argument");
    AttributeList AL = NF->getAttributes();
    for (unsigned I = 0; I <= LastPreloadIndex; ++I) {
      AL = AL.addParamAttributes(Ctx, NFArg->getArgNo(), AB);
      NFArg++->setName(getHiddenArgName(HiddenArg(I)));
    }

    NF->setAttributes(AL);
    F.replaceAllUsesWith(NF);

    return NF;
  }

public:
  PreloadKernelArgInfo(Function &F, const GCNSubtarget &ST) : F(F), ST(ST) {
    setInitialFreeUserSGPRsCount();
  }

  // Returns the maximum number of user SGPRs that we have available to preload
  // arguments.
  void setInitialFreeUserSGPRsCount() {
    GCNUserSGPRUsageInfo UserSGPRInfo(F, ST);
    NumFreeUserSGPRs = UserSGPRInfo.getNumFreeUserSGPRs();
  }

  bool canPreloadKernArgAtOffset(uint64_t ExplicitArgOffset) {
    return ExplicitArgOffset <= NumFreeUserSGPRs * 4;
  }

  // Try to allocate SGPRs to preload hidden kernel arguments.
  void
  tryAllocHiddenArgPreloadSGPRs(uint64_t ImplicitArgsBaseOffset,
                                SmallVectorImpl<Function *> &FunctionsToErase) {
    Function *ImplicitArgPtr = Intrinsic::getDeclarationIfExists(
        F.getParent(), Intrinsic::amdgcn_implicitarg_ptr);
    if (!ImplicitArgPtr)
      return;

    const DataLayout &DL = F.getParent()->getDataLayout();
    // Pair is the load and the load offset.
    SmallVector<std::pair<LoadInst *, unsigned>, 4> ImplicitArgLoads;
    for (auto *U : ImplicitArgPtr->users()) {
      Instruction *CI = dyn_cast<Instruction>(U);
      if (!CI || CI->getParent()->getParent() != &F)
        continue;

      for (auto *U : CI->users()) {
        int64_t Offset = 0;
        auto *Load = dyn_cast<LoadInst>(U); // Load from ImplicitArgPtr?
        if (!Load) {
          if (GetPointerBaseWithConstantOffset(U, Offset, DL) != CI)
            continue;

          Load = dyn_cast<LoadInst>(*U->user_begin()); // Load from GEP?
        }

        if (!Load || !Load->isSimple())
          continue;

        // FIXME: Expand handle merged loads.
        LLVMContext &Ctx = F.getParent()->getContext();
        Type *LoadTy = Load->getType();
        HiddenArg HA = getHiddenArgFromOffset(Offset);
        if (HA == END_HIDDEN_ARGS || LoadTy != getHiddenArgType(Ctx, HA))
          continue;

        ImplicitArgLoads.push_back(std::make_pair(Load, Offset));
      }
    }

    if (ImplicitArgLoads.empty())
      return;

    // Allocate loads in order of offset. We need to be sure that the implicit
    // argument can actually be preloaded.
    std::sort(ImplicitArgLoads.begin(), ImplicitArgLoads.end(), less_second());

    // If we fail to preload any implicit argument we know we don't have SGPRs
    // to preload any subsequent ones with larger offsets. Find the first
    // argument that we cannot preload.
    auto *PreloadEnd = llvm::find_if(
        ImplicitArgLoads, [&](const std::pair<LoadInst *, unsigned> &Load) {
          unsigned LoadSize = DL.getTypeStoreSize(Load.first->getType());
          unsigned LoadOffset = Load.second;
          if (!canPreloadKernArgAtOffset(LoadOffset + LoadSize +
                                         ImplicitArgsBaseOffset))
            return true;

          return false;
        });

    if (PreloadEnd == ImplicitArgLoads.begin())
      return;

    unsigned LastHiddenArgIndex = getHiddenArgFromOffset(PreloadEnd[-1].second);
    Function *NF = cloneFunctionWithPreloadImplicitArgs(LastHiddenArgIndex);
    assert(NF);
    FunctionsToErase.push_back(&F);
    for (const auto *I = ImplicitArgLoads.begin(); I != PreloadEnd; ++I) {
      LoadInst *LoadInst = I->first;
      unsigned LoadOffset = I->second;
      unsigned HiddenArgIndex = getHiddenArgFromOffset(LoadOffset);
      unsigned Index = NF->arg_size() - LastHiddenArgIndex + HiddenArgIndex - 1;
      Argument *Arg = NF->getArg(Index);
      LoadInst->replaceAllUsesWith(Arg);
    }
  }
};

} // end anonymous namespace

char AMDGPUPreloadKernelArgumentsLegacy::ID = 0;

INITIALIZE_PASS(AMDGPUPreloadKernelArgumentsLegacy, DEBUG_TYPE,
                "AMDGPU Preload Kernel Arguments", false, false)

ModulePass *
llvm::createAMDGPUPreloadKernelArgumentsLegacyPass(const TargetMachine *TM) {
  return new AMDGPUPreloadKernelArgumentsLegacy(
      static_cast<const GCNTargetMachine *>(TM));
}

AMDGPUPreloadKernelArgumentsLegacy::AMDGPUPreloadKernelArgumentsLegacy(
    const GCNTargetMachine *TM)
    : ModulePass(ID), TM(TM) {}

static bool markKernelArgsAsInreg(Module &M, const TargetMachine &TM) {
  if (!EnableKernargPreload)
    return false;

  SmallVector<Function *, 4> FunctionsToErase;
  bool Changed = false;
  for (auto &F : M) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    if (!ST.hasKernargPreload() ||
        F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
      continue;

    PreloadKernelArgInfo PreloadInfo(F, ST);
    uint64_t ExplicitArgOffset = 0;
    const DataLayout &DL = F.getDataLayout();
    const uint64_t BaseOffset = ST.getExplicitKernelArgOffset();
    unsigned NumPreloadsRequested = KernargPreloadCount;
    unsigned NumPreloadedExplicitArgs = 0;
    for (Argument &Arg : F.args()) {
      // Avoid incompatible attributes and guard against running this pass
      // twice.
      //
      // TODO: Preload byref kernel arguments
      if (Arg.hasByRefAttr() || Arg.hasNestAttr() ||
          Arg.hasAttribute("amdgpu-hidden-argument"))
        break;

      // Inreg may be pre-existing on some arguments, try to preload these.
      if (NumPreloadsRequested == 0 && !Arg.hasInRegAttr())
        break;

      // FIXME: Preload aggregates.
      if (Arg.getType()->isAggregateType())
        break;

      Type *ArgTy = Arg.getType();
      Align ABITypeAlign = DL.getABITypeAlign(ArgTy);
      uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);
      ExplicitArgOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + AllocSize;

      if (!PreloadInfo.canPreloadKernArgAtOffset(ExplicitArgOffset))
        break;

      Arg.addAttr(Attribute::InReg);
      NumPreloadedExplicitArgs++;
      if (NumPreloadsRequested > 0)
        NumPreloadsRequested--;
    }

    // Only try preloading hidden arguments if we can successfully preload the
    // last explicit argument.
    if (NumPreloadedExplicitArgs == F.arg_size()) {
      uint64_t ImplicitArgsBaseOffset =
          alignTo(ExplicitArgOffset, ST.getAlignmentForImplicitArgPtr()) +
          BaseOffset;
      PreloadInfo.tryAllocHiddenArgPreloadSGPRs(ImplicitArgsBaseOffset,
                                                FunctionsToErase);
    }

    Changed |= NumPreloadedExplicitArgs > 0;
  }

  // Erase cloned functions if we needed to update the kernel signature to
  // support preloading hidden kernel arguments.
  for (auto *F : FunctionsToErase)
    F->eraseFromParent();

  return Changed;
}

bool AMDGPUPreloadKernelArgumentsLegacy::runOnModule(Module &M) {
  if (skipModule(M) || !TM)
    return false;

  return markKernelArgsAsInreg(M, *TM);
}

PreservedAnalyses
AMDGPUPreloadKernelArgumentsPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool Changed = markKernelArgsAsInreg(M, TM);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
