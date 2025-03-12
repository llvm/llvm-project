//===-- AMDGPULowerKernelArguments.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass replaces accesses to kernel arguments with loads from
/// offsets from the kernarg base pointer.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-lower-kernel-arguments"

using namespace llvm;

namespace {

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
    if (HA < END_HIDDEN_ARGS) {
      return HiddenArgs[HA].Name;
    }
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
    NF->setIsNewDbgInfoFormat(F.IsNewDbgInfoFormat);

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
    F.setCallingConv(CallingConv::C);

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

  bool tryAllocPreloadSGPRs(unsigned AllocSize, uint64_t ArgOffset,
                            uint64_t LastExplicitArgOffset) {
    //  Check if this argument may be loaded into the same register as the
    //  previous argument.
    if (ArgOffset - LastExplicitArgOffset < 4 &&
        !isAligned(Align(4), ArgOffset))
      return true;

    // Pad SGPRs for kernarg alignment.
    ArgOffset = alignDown(ArgOffset, 4);
    unsigned Padding = ArgOffset - LastExplicitArgOffset;
    unsigned PaddingSGPRs = alignTo(Padding, 4) / 4;
    unsigned NumPreloadSGPRs = alignTo(AllocSize, 4) / 4;
    if (NumPreloadSGPRs + PaddingSGPRs > NumFreeUserSGPRs)
      return false;

    NumFreeUserSGPRs -= (NumPreloadSGPRs + PaddingSGPRs);
    return true;
  }

  // Try to allocate SGPRs to preload implicit kernel arguments.
  void tryAllocImplicitArgPreloadSGPRs(uint64_t ImplicitArgsBaseOffset,
                                       uint64_t LastExplicitArgOffset,
                                       IRBuilder<> &Builder) {
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

        // FIXME: Expand to handle 64-bit implicit args and large merged loads.
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
    auto *PreloadEnd = std::find_if(
        ImplicitArgLoads.begin(), ImplicitArgLoads.end(),
        [&](const std::pair<LoadInst *, unsigned> &Load) {
          unsigned LoadSize = DL.getTypeStoreSize(Load.first->getType());
          unsigned LoadOffset = Load.second;
          if (!tryAllocPreloadSGPRs(LoadSize,
                                    LoadOffset + ImplicitArgsBaseOffset,
                                    LastExplicitArgOffset))
            return true;

          LastExplicitArgOffset =
              ImplicitArgsBaseOffset + LoadOffset + LoadSize;
          return false;
        });

    if (PreloadEnd == ImplicitArgLoads.begin())
      return;

    unsigned LastHiddenArgIndex = getHiddenArgFromOffset(PreloadEnd[-1].second);
    Function *NF = cloneFunctionWithPreloadImplicitArgs(LastHiddenArgIndex);
    assert(NF);
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

class AMDGPULowerKernelArguments : public FunctionPass {
public:
  static char ID;

  AMDGPULowerKernelArguments() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
 }
};

} // end anonymous namespace

// skip allocas
static BasicBlock::iterator getInsertPt(BasicBlock &BB) {
  BasicBlock::iterator InsPt = BB.getFirstInsertionPt();
  for (BasicBlock::iterator E = BB.end(); InsPt != E; ++InsPt) {
    AllocaInst *AI = dyn_cast<AllocaInst>(&*InsPt);

    // If this is a dynamic alloca, the value may depend on the loaded kernargs,
    // so loads will need to be inserted before it.
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  return InsPt;
}

static bool lowerKernelArguments(Function &F, const TargetMachine &TM) {
  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty())
    return false;

  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  LLVMContext &Ctx = F.getParent()->getContext();
  const DataLayout &DL = F.getDataLayout();
  BasicBlock &EntryBlock = *F.begin();
  IRBuilder<> Builder(&EntryBlock, getInsertPt(EntryBlock));

  const Align KernArgBaseAlign(16); // FIXME: Increase if necessary
  const uint64_t BaseOffset = ST.getExplicitKernelArgOffset();

  Align MaxAlign;
  // FIXME: Alignment is broken with explicit arg offset.;
  const uint64_t TotalKernArgSize = ST.getKernArgSegmentSize(F, MaxAlign);
  if (TotalKernArgSize == 0)
    return false;

  CallInst *KernArgSegment =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_kernarg_segment_ptr, {}, {},
                              nullptr, F.getName() + ".kernarg.segment");
  KernArgSegment->addRetAttr(Attribute::NonNull);
  KernArgSegment->addRetAttr(
      Attribute::getWithDereferenceableBytes(Ctx, TotalKernArgSize));

  uint64_t ExplicitArgOffset = 0;
  // Preloaded kernel arguments must be sequential.
  bool InPreloadSequence = true;
  PreloadKernelArgInfo PreloadInfo(F, ST);

  for (Argument &Arg : F.args()) {
    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    MaybeAlign ParamAlign = IsByRef ? Arg.getParamAlign() : std::nullopt;
    Align ABITypeAlign = DL.getValueOrABITypeAlignment(ParamAlign, ArgTy);

    uint64_t Size = DL.getTypeSizeInBits(ArgTy);
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);

    uint64_t EltOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + BaseOffset;
    uint64_t LastExplicitArgOffset = ExplicitArgOffset;
    ExplicitArgOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + AllocSize;

    // Guard against the situation where hidden arguments have already been
    // lowered and added to the kernel function signiture, i.e. in a situation
    // where this pass has run twice.
    if (Arg.hasAttribute("amdgpu-hidden-argument"))
      break;

    // Try to preload this argument into user SGPRs.
    if (Arg.hasInRegAttr() && InPreloadSequence && ST.hasKernargPreload() &&
        !Arg.getType()->isAggregateType())
      if (PreloadInfo.tryAllocPreloadSGPRs(AllocSize, EltOffset,
                                           LastExplicitArgOffset))
        continue;

    InPreloadSequence = false;

    if (Arg.use_empty())
      continue;

    // If this is byval, the loads are already explicit in the function. We just
    // need to rewrite the pointer values.
    if (IsByRef) {
      Value *ArgOffsetPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, EltOffset,
          Arg.getName() + ".byval.kernarg.offset");

      Value *CastOffsetPtr =
          Builder.CreateAddrSpaceCast(ArgOffsetPtr, Arg.getType());
      Arg.replaceAllUsesWith(CastOffsetPtr);
      continue;
    }

    if (PointerType *PT = dyn_cast<PointerType>(ArgTy)) {
      // FIXME: Hack. We rely on AssertZext to be able to fold DS addressing
      // modes on SI to know the high bits are 0 so pointer adds don't wrap. We
      // can't represent this with range metadata because it's only allowed for
      // integer types.
      if ((PT->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
           PT->getAddressSpace() == AMDGPUAS::REGION_ADDRESS) &&
          !ST.hasUsableDSOffset())
        continue;

      // FIXME: We can replace this with equivalent alias.scope/noalias
      // metadata, but this appears to be a lot of work.
      if (Arg.hasNoAliasAttr())
        continue;
    }

    auto *VT = dyn_cast<FixedVectorType>(ArgTy);
    bool IsV3 = VT && VT->getNumElements() == 3;
    bool DoShiftOpt = Size < 32 && !ArgTy->isAggregateType();

    VectorType *V4Ty = nullptr;

    int64_t AlignDownOffset = alignDown(EltOffset, 4);
    int64_t OffsetDiff = EltOffset - AlignDownOffset;
    Align AdjustedAlign = commonAlignment(
        KernArgBaseAlign, DoShiftOpt ? AlignDownOffset : EltOffset);

    Value *ArgPtr;
    Type *AdjustedArgTy;
    if (DoShiftOpt) { // FIXME: Handle aggregate types
      // Since we don't have sub-dword scalar loads, avoid doing an extload by
      // loading earlier than the argument address, and extracting the relevant
      // bits.
      // TODO: Update this for GFX12 which does have scalar sub-dword loads.
      //
      // Additionally widen any sub-dword load to i32 even if suitably aligned,
      // so that CSE between different argument loads works easily.
      ArgPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, AlignDownOffset,
          Arg.getName() + ".kernarg.offset.align.down");
      AdjustedArgTy = Builder.getInt32Ty();
    } else {
      ArgPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, EltOffset,
          Arg.getName() + ".kernarg.offset");
      AdjustedArgTy = ArgTy;
    }

    if (IsV3 && Size >= 32) {
      V4Ty = FixedVectorType::get(VT->getElementType(), 4);
      // Use the hack that clang uses to avoid SelectionDAG ruining v3 loads
      AdjustedArgTy = V4Ty;
    }

    LoadInst *Load =
        Builder.CreateAlignedLoad(AdjustedArgTy, ArgPtr, AdjustedAlign);
    Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));

    MDBuilder MDB(Ctx);

    if (Arg.hasAttribute(Attribute::NoUndef))
      Load->setMetadata(LLVMContext::MD_noundef, MDNode::get(Ctx, {}));

    if (Arg.hasAttribute(Attribute::Range)) {
      const ConstantRange &Range =
          Arg.getAttribute(Attribute::Range).getValueAsConstantRange();
      Load->setMetadata(LLVMContext::MD_range,
                        MDB.createRange(Range.getLower(), Range.getUpper()));
    }

    if (isa<PointerType>(ArgTy)) {
      if (Arg.hasNonNullAttr())
        Load->setMetadata(LLVMContext::MD_nonnull, MDNode::get(Ctx, {}));

      uint64_t DerefBytes = Arg.getDereferenceableBytes();
      if (DerefBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable,
          MDNode::get(Ctx,
                      MDB.createConstant(
                        ConstantInt::get(Builder.getInt64Ty(), DerefBytes))));
      }

      uint64_t DerefOrNullBytes = Arg.getDereferenceableOrNullBytes();
      if (DerefOrNullBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable_or_null,
          MDNode::get(Ctx,
                      MDB.createConstant(ConstantInt::get(Builder.getInt64Ty(),
                                                          DerefOrNullBytes))));
      }

      if (MaybeAlign ParamAlign = Arg.getParamAlign()) {
        Load->setMetadata(
            LLVMContext::MD_align,
            MDNode::get(Ctx, MDB.createConstant(ConstantInt::get(
                                 Builder.getInt64Ty(), ParamAlign->value()))));
      }
    }

    // TODO: Convert noalias arg to !noalias

    if (DoShiftOpt) {
      Value *ExtractBits = OffsetDiff == 0 ?
        Load : Builder.CreateLShr(Load, OffsetDiff * 8);

      IntegerType *ArgIntTy = Builder.getIntNTy(Size);
      Value *Trunc = Builder.CreateTrunc(ExtractBits, ArgIntTy);
      Value *NewVal = Builder.CreateBitCast(Trunc, ArgTy,
                                            Arg.getName() + ".load");
      Arg.replaceAllUsesWith(NewVal);
    } else if (IsV3) {
      Value *Shuf = Builder.CreateShuffleVector(Load, ArrayRef<int>{0, 1, 2},
                                                Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Shuf);
    } else {
      Load->setName(Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Load);
    }
  }

  KernArgSegment->addRetAttr(
      Attribute::getWithAlignment(Ctx, std::max(KernArgBaseAlign, MaxAlign)));

  if (InPreloadSequence) {
    uint64_t ImplicitArgsBaseOffset =
        alignTo(ExplicitArgOffset, ST.getAlignmentForImplicitArgPtr()) +
        BaseOffset;
    PreloadInfo.tryAllocImplicitArgPreloadSGPRs(ImplicitArgsBaseOffset,
                                                ExplicitArgOffset, Builder);
  }

  return true;
}

bool AMDGPULowerKernelArguments::runOnFunction(Function &F) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  return lowerKernelArguments(F, TM);
}

INITIALIZE_PASS_BEGIN(AMDGPULowerKernelArguments, DEBUG_TYPE,
                      "AMDGPU Lower Kernel Arguments", false, false)
INITIALIZE_PASS_END(AMDGPULowerKernelArguments, DEBUG_TYPE, "AMDGPU Lower Kernel Arguments",
                    false, false)

char AMDGPULowerKernelArguments::ID = 0;

FunctionPass *llvm::createAMDGPULowerKernelArgumentsPass() {
  return new AMDGPULowerKernelArguments();
}

PreservedAnalyses
AMDGPULowerKernelArgumentsPass::run(Function &F, FunctionAnalysisManager &AM) {
  bool Changed = lowerKernelArguments(F, TM);
  if (Changed) {
    // TODO: Preserves a lot more.
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}
