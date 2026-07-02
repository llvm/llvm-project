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
///
/// This pass also handles splitting of byref struct kernel arguments into
/// scalar arguments when doing so would allow them to be preloaded. The
/// splitting only occurs if the split arguments can fit in available SGPRs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
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

static cl::opt<bool> EnableKernargLayoutChange(
    "amdgpu-kernarg-layout-change",
    cl::desc("Allow changing kernel argument segment layout when splitting "
             "byref structs (remove unused fields, reorder for packing). "
             "When disabled (default), all struct fields are preserved in "
             "their original order."),
    cl::init(false));

namespace {

//===----------------------------------------------------------------------===//
// Kernel Argument Splitting Logic
//
// The following functions handle splitting of byref struct kernel arguments
// into scalar arguments. This enables preloading of struct fields that would
// otherwise not be preloadable due to the byref attribute.
//===----------------------------------------------------------------------===//

// Attribute name for tracking original argument index and offset
static constexpr StringRef OriginalArgAttr = "amdgpu-original-arg";

// Prefix for backup declaration of original kernel (used for metadata
// generation)
static constexpr StringRef OriginalKernelPrefix = "__amdgpu_orig_kernel_";

// Attribute to store the name of the backup declaration
static constexpr StringRef OriginalKernelAttr = "amdgpu-original-kernel";

static bool parseOriginalArgAttribute(StringRef S, unsigned &RootIdx,
                                      uint64_t &BaseOff) {
  auto Parts = S.split(':');
  if (Parts.second.empty())
    return false;
  if (Parts.first.getAsInteger(10, RootIdx))
    return false;
  if (Parts.second.getAsInteger(10, BaseOff))
    return false;
  return true;
}

/// Traverses all users of an argument to check if it's suitable for
/// splitting. A suitable argument is only used by a chain of
/// GEPs that terminate in LoadInsts.
static bool
areArgUsersValidForSplit(Argument &Arg, SmallVectorImpl<LoadInst *> &Loads,
                         SmallVectorImpl<GetElementPtrInst *> &GEPs) {
  SmallVector<User *, 16> Worklist(Arg.user_begin(), Arg.user_end());
  SetVector<User *> Visited;

  while (!Worklist.empty()) {
    User *U = Worklist.pop_back_val();
    if (!Visited.insert(U))
      continue;

    if (auto *LI = dyn_cast<LoadInst>(U)) {
      Loads.push_back(LI);
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      GEPs.push_back(GEP);
      for (User *GEPUser : GEP->users())
        Worklist.push_back(GEPUser);
    } else {
      return false;
    }
  }

  const DataLayout &DL = Arg.getParent()->getParent()->getDataLayout();
  for (const LoadInst *LI : Loads) {
    APInt Offset(DL.getPointerSizeInBits(), 0);
    const Value *Base =
        LI->getPointerOperand()->stripAndAccumulateConstantOffsets(
            DL, Offset, /*AllowNonInbounds=*/false);
    if (Base != &Arg)
      return false;
  }

  return true;
}

/// Information about a struct field to be flattened into a scalar argument.
struct FieldInfo {
  Type *Ty = nullptr;
  uint64_t Offset = 0;
  LoadInst *Load = nullptr; // nullptr if field is unused
};

/// Recursively collect all leaf (scalar) fields from a type with their offsets.
/// This flattens nested structs and arrays into individual scalar fields.
static void collectLeafFields(Type *Ty, const DataLayout &DL,
                              uint64_t BaseOffset,
                              SmallVectorImpl<FieldInfo> &Fields) {
  if (auto *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(STy);
    for (unsigned I = 0; I < STy->getNumElements(); ++I) {
      Type *ElemTy = STy->getElementType(I);
      uint64_t ElemOffset = BaseOffset + SL->getElementOffset(I);
      collectLeafFields(ElemTy, DL, ElemOffset, Fields);
    }
  } else if (auto *ATy = dyn_cast<ArrayType>(Ty)) {
    Type *ElemTy = ATy->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (uint64_t I = 0; I < ATy->getNumElements(); ++I) {
      collectLeafFields(ElemTy, DL, BaseOffset + I * ElemSize, Fields);
    }
  } else {
    // Leaf type (scalar, vector, pointer, etc.)
    Fields.push_back({Ty, BaseOffset, nullptr});
  }
}

/// Check if split arguments can be preloaded into SGPRs.
/// This calculates the new arg layout size after splitting and checks if it
/// fits in available user SGPRs.
static bool canPreloadSplitArgs(
    Function &F, const GCNSubtarget &ST,
    const DenseMap<Argument *, SmallVector<FieldInfo, 8>> &ArgToFieldsMap) {
  GCNUserSGPRUsageInfo UserSGPRInfo(F, ST);
  unsigned NumFreeUserSGPRs = UserSGPRInfo.getNumFreeUserSGPRs();
  uint64_t AvailableBytes = NumFreeUserSGPRs * 4;

  const DataLayout &DL = F.getParent()->getDataLayout();
  uint64_t NewArgOffset = 0;

  // Calculate the new arg layout size after splitting
  for (Argument &Arg : F.args()) {
    auto It = ArgToFieldsMap.find(&Arg);
    if (It != ArgToFieldsMap.end()) {
      // This arg will be split - add sizes of replacement scalar args
      for (const FieldInfo &FI : It->second) {
        Align ABITypeAlign = DL.getABITypeAlign(FI.Ty);
        uint64_t AllocSize = DL.getTypeAllocSize(FI.Ty);
        NewArgOffset = alignTo(NewArgOffset, ABITypeAlign) + AllocSize;
      }
    } else {
      // This arg is not split - keep original size
      Type *ArgTy = Arg.getType();
      if (Arg.hasByRefAttr())
        ArgTy = Arg.getParamByRefType();
      Align ABITypeAlign = DL.getABITypeAlign(ArgTy);
      uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);
      NewArgOffset = alignTo(NewArgOffset, ABITypeAlign) + AllocSize;
    }
  }

  return NewArgOffset <= AvailableBytes;
}

/// Try to split byref struct kernel arguments into scalar arguments.
/// Returns the new function with split arguments, or nullptr if no split
/// was performed. If a new function is returned, the original function F
/// has been erased and should not be used.
///
/// When EnableKernargLayoutChange is false (default), ALL struct fields are
/// preserved in their original order (recursively flattening nested structs),
/// maintaining the kernel argument segment layout. Unused fields become dead
/// arguments.
///
/// When EnableKernargLayoutChange is true, only used fields are kept and
/// the layout may change.
static Function *trySplitKernelArguments(Function &F, const GCNSubtarget &ST) {
  if (!ST.hasKernargPreload())
    return nullptr;

  if (F.isDeclaration() || !AMDGPU::isKernel(F.getCallingConv()) ||
      F.arg_empty())
    return nullptr;

  const DataLayout &DL = F.getParent()->getDataLayout();

  // Mappings from new arg index to original arg: (NewArgIdx, OrigArgIdx,
  // Offset)
  SmallVector<std::tuple<unsigned, unsigned, uint64_t>, 8> NewArgMappings;
  DenseMap<Argument *, SmallVector<LoadInst *, 8>> ArgToLoadsMap;
  DenseMap<Argument *, SmallVector<GetElementPtrInst *, 8>> ArgToGEPsMap;
  // Maps struct arg to field info (type, offset, associated load if any)
  DenseMap<Argument *, SmallVector<FieldInfo, 8>> ArgToFieldsMap;
  SmallVector<Argument *, 8> StructArgs;
  SmallVector<Type *, 8> NewArgTypes;

  unsigned OriginalArgIndex = 0;
  unsigned NewArgIndex = 0;
  auto HandlePassthroughArg = [&](Argument &Arg) {
    NewArgTypes.push_back(Arg.getType());
    if (!Arg.hasAttribute(OriginalArgAttr) && NewArgIndex != OriginalArgIndex)
      NewArgMappings.emplace_back(NewArgIndex, OriginalArgIndex, 0);
    ++NewArgIndex;
    ++OriginalArgIndex;
  };

  for (Argument &Arg : F.args()) {
    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT || !Arg.hasByRefAttr()) {
      HandlePassthroughArg(Arg);
      continue;
    }

    StructType *STy = dyn_cast<StructType>(Arg.getParamByRefType());
    if (!STy) {
      HandlePassthroughArg(Arg);
      continue;
    }

    // Collect loads from this struct argument
    SmallVector<LoadInst *, 8> Loads;
    SmallVector<GetElementPtrInst *, 8> GEPs;

    // Check if all users are valid for splitting (GEPs + loads)
    bool HasValidUsers =
        Arg.use_empty() || areArgUsersValidForSplit(Arg, Loads, GEPs);
    if (!HasValidUsers) {
      HandlePassthroughArg(Arg);
      continue;
    }

    // Helper to get load offset. Returns std::nullopt if offset can't be
    // computed (e.g., variable-index GEP).
    auto GetLoadOffset = [&](LoadInst *LI) -> std::optional<uint64_t> {
      Value *Ptr = LI->getPointerOperand();
      // Direct load from argument (offset 0)
      if (Ptr == &Arg)
        return 0;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        APInt OffsetAPInt(DL.getPointerSizeInBits(), 0);
        if (GEP->accumulateConstantOffset(DL, OffsetAPInt))
          return OffsetAPInt.getZExtValue();
      }
      return std::nullopt;
    };

    unsigned RootIdx = OriginalArgIndex;
    uint64_t BaseOffset = 0;

    if (Arg.hasAttribute(OriginalArgAttr)) {
      Attribute Attr = F.getAttributeAtIndex(OriginalArgIndex, OriginalArgAttr);
      (void)parseOriginalArgAttribute(Attr.getValueAsString(), RootIdx,
                                      BaseOffset);
    }

    // Build maps from offset to load and load to offset. Skip splitting if any
    // load has a variable-index GEP (can't compute constant offset).
    DenseMap<uint64_t, LoadInst *> OffsetToLoad;
    DenseMap<LoadInst *, uint64_t> LoadToOffset;
    bool HasVariableIndexLoad = false;
    for (LoadInst *LI : Loads) {
      std::optional<uint64_t> Off = GetLoadOffset(LI);
      if (!Off) {
        HasVariableIndexLoad = true;
        break;
      }
      OffsetToLoad[*Off] = LI;
      LoadToOffset[LI] = *Off;
    }

    if (HasVariableIndexLoad) {
      LLVM_DEBUG(dbgs() << "Skipping split for " << F.getName()
                        << ": load with variable-index GEP\n");
      HandlePassthroughArg(Arg);
      continue;
    }

    StructArgs.push_back(&Arg);
    ArgToLoadsMap[&Arg] = Loads;
    ArgToGEPsMap[&Arg] = GEPs;

    SmallVector<FieldInfo, 8> Fields;

    if (EnableKernargLayoutChange) {
      // Layout change allowed: only keep used fields, sorted by offset
      llvm::sort(Loads, [&](LoadInst *A, LoadInst *B) {
        return LoadToOffset[A] < LoadToOffset[B];
      });

      for (LoadInst *LI : Loads) {
        uint64_t LocalOff = LoadToOffset[LI];
        Fields.push_back({LI->getType(), LocalOff, LI});
        NewArgTypes.push_back(LI->getType());
        uint64_t FinalOff = BaseOffset + LocalOff;
        NewArgMappings.emplace_back(NewArgIndex, RootIdx, FinalOff);
        ++NewArgIndex;
      }
    } else {
      // Layout preserved: keep ALL leaf fields in original struct order
      // Recursively flatten nested structs
      collectLeafFields(STy, DL, /*BaseOffset=*/0, Fields);

      // Build a map from (offset, type) to field index for matching
      DenseMap<std::pair<uint64_t, Type *>, unsigned> OffsetTypeToField;
      for (unsigned I = 0; I < Fields.size(); ++I)
        OffsetTypeToField[{Fields[I].Offset, Fields[I].Ty}] = I;

      // Verify all loads can be matched to a leaf field
      bool AllLoadsMatch = true;
      for (LoadInst *LI : Loads) {
        uint64_t Off = LoadToOffset[LI];
        auto Key = std::make_pair(Off, LI->getType());
        if (!OffsetTypeToField.count(Key)) {
          AllLoadsMatch = false;
          break;
        }
      }

      if (!AllLoadsMatch) {
        LLVM_DEBUG(dbgs() << "Skipping split for " << F.getName()
                          << ": load type doesn't match leaf field type\n");
        // Undo: remove from StructArgs, restore passthrough
        StructArgs.pop_back();
        ArgToLoadsMap.erase(&Arg);
        ArgToGEPsMap.erase(&Arg);
        HandlePassthroughArg(Arg);
        continue;
      }

      // Associate loads with their corresponding fields
      for (LoadInst *LI : Loads) {
        uint64_t Off = LoadToOffset[LI];
        auto Key = std::make_pair(Off, LI->getType());
        unsigned FieldIdx = OffsetTypeToField[Key];
        Fields[FieldIdx].Load = LI;
      }

      for (const FieldInfo &FI : Fields) {
        NewArgTypes.push_back(FI.Ty);
        uint64_t FinalOff = BaseOffset + FI.Offset;
        NewArgMappings.emplace_back(NewArgIndex, RootIdx, FinalOff);
        ++NewArgIndex;
      }
    }

    ArgToFieldsMap[&Arg] = Fields;
    ++OriginalArgIndex;
  }

  if (StructArgs.empty())
    return nullptr;

  if (!canPreloadSplitArgs(F, ST, ArgToFieldsMap)) {
    LLVM_DEBUG(dbgs() << "Skipping split for " << F.getName()
                      << ": split args would not fit in preload SGPRs\n");
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "Splitting kernel arguments for " << F.getName()
                    << "\n");

  // Create new function
  AttributeList OldAttrs = F.getAttributes();
  AttributeSet FnAttrs = OldAttrs.getFnAttrs();
  AttributeSet RetAttrs = OldAttrs.getRetAttrs();

  FunctionType *NewFT =
      FunctionType::get(F.getReturnType(), NewArgTypes, F.isVarArg());
  Function *NewF =
      Function::Create(NewFT, F.getLinkage(), F.getAddressSpace(), F.getName());
  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);
  NewF->takeName(&F);
  NewF->setVisibility(F.getVisibility());
  if (F.hasComdat())
    NewF->setComdat(F.getComdat());
  NewF->setDSOLocal(F.isDSOLocal());
  NewF->setUnnamedAddr(F.getUnnamedAddr());
  NewF->setCallingConv(F.getCallingConv());

  SmallVector<AttributeSet, 8> NewArgAttrSets;
  NewArgIndex = 0;
  for (Argument &Arg : F.args()) {
    if (ArgToFieldsMap.count(&Arg)) {
      for ([[maybe_unused]] const FieldInfo &FI : ArgToFieldsMap[&Arg]) {
        NewArgAttrSets.push_back(AttributeSet());
        ++NewArgIndex;
      }
    } else {
      AttributeSet ArgAttrs = OldAttrs.getParamAttrs(Arg.getArgNo());
      NewArgAttrSets.push_back(ArgAttrs);
      ++NewArgIndex;
    }
  }

  AttributeList NewAttrList =
      AttributeList::get(F.getContext(), FnAttrs, RetAttrs, NewArgAttrSets);
  NewF->setAttributes(NewAttrList);

  // In layout-changing mode, add original-arg attributes so CLR can map
  // split args back. In layout-preserving mode, we use a backup declaration
  // instead, so no per-arg attributes needed.
  if (EnableKernargLayoutChange) {
    for (const auto &Info : NewArgMappings) {
      unsigned NewArgIdx, RootArgIdx;
      uint64_t Offset;
      std::tie(NewArgIdx, RootArgIdx, Offset) = Info;
      NewF->addParamAttr(
          NewArgIdx,
          Attribute::get(NewF->getContext(), OriginalArgAttr,
                         (Twine(RootArgIdx) + ":" + Twine(Offset)).str()));
    }
  }

  LLVM_DEBUG(dbgs() << "New function signature:\n" << *NewF << '\n');

  NewF->splice(NewF->begin(), &F);

  DenseMap<Value *, Value *> VMap;
  auto NewArgIt = NewF->arg_begin();
  for (Argument &Arg : F.args()) {
    if (ArgToFieldsMap.contains(&Arg)) {
      for (const FieldInfo &FI : ArgToFieldsMap[&Arg]) {
        Value *NewArg = &*NewArgIt++;
        if (FI.Load) {
          // This field has an associated load - map it
          NewArg->takeName(FI.Load);
          if (isa<PointerType>(NewArg->getType()) &&
              isa<PointerType>(FI.Load->getType())) {
            IRBuilder<> Builder(FI.Load);
            Value *CastedArg = Builder.CreatePointerBitCastOrAddrSpaceCast(
                NewArg, FI.Load->getType());
            VMap[FI.Load] = CastedArg;
          } else {
            VMap[FI.Load] = NewArg;
          }
        }
        // If FI.Load is null, this is an unused field - arg exists but unused
      }
      PoisonValue *PoisonArg = PoisonValue::get(Arg.getType());
      Arg.replaceAllUsesWith(PoisonArg);
    } else {
      NewArgIt->takeName(&Arg);
      Value *NewArg = &*NewArgIt;
      if (isa<PointerType>(NewArg->getType()) &&
          isa<PointerType>(Arg.getType())) {
        IRBuilder<> Builder(&*NewF->begin()->begin());
        Value *CastedArg =
            Builder.CreatePointerBitCastOrAddrSpaceCast(NewArg, Arg.getType());
        Arg.replaceAllUsesWith(CastedArg);
      } else {
        Arg.replaceAllUsesWith(NewArg);
      }
      ++NewArgIt;
    }
  }

  for (auto &Entry : ArgToLoadsMap) {
    for (LoadInst *LI : Entry.second) {
      Value *NewArg = VMap.lookup(LI);
      assert(NewArg && "Load not mapped to new argument - did we miss a "
                       "variable-index GEP check?");
      LI->replaceAllUsesWith(NewArg);
      LI->eraseFromParent();
    }
  }

  for (auto &Entry : ArgToGEPsMap) {
    for (GetElementPtrInst *GEP : Entry.second) {
      GEP->replaceAllUsesWith(PoisonValue::get(GEP->getType()));
      GEP->eraseFromParent();
    }
  }

  LLVM_DEBUG(dbgs() << "Function after splitting:\n" << *NewF << '\n');

  // In layout-preserving mode, create a backup declaration of the original
  // function. This backup is used by the metadata streamer to emit the
  // original kernel argument metadata (so CLR doesn't need patching).
  if (!EnableKernargLayoutChange) {
    std::string BackupName =
        (Twine(OriginalKernelPrefix) + NewF->getName()).str();
    // Create a declaration with the original signature (no body needed).
    // This is used purely for metadata extraction, not code generation.
    Function *BackupF =
        Function::Create(F.getFunctionType(), GlobalValue::ExternalLinkage,
                         F.getAddressSpace(), BackupName, F.getParent());
    // Copy function attributes for metadata extraction
    BackupF->setAttributes(F.getAttributes());
    BackupF->copyMetadata(&F, 0);
    // Use C calling convention so it's not processed as a kernel
    BackupF->setCallingConv(CallingConv::C);

    // Store the backup name in the new function
    NewF->addFnAttr(OriginalKernelAttr, BackupName);

    LLVM_DEBUG(dbgs() << "Created backup declaration: " << BackupName << '\n');
  }

  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();

  return NewF;
}

//===----------------------------------------------------------------------===//
// Preload Kernel Arguments Logic
//===----------------------------------------------------------------------===//

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
    LLVMContext &Ctx = F.getContext();
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
      if (!CI || CI->getFunction() != &F)
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
        LLVMContext &Ctx = F.getContext();
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

  // First, collect functions to process (split may modify the function list)
  SmallVector<Function *, 16> FunctionsToProcess;
  for (auto &F : M) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    if (!ST.hasKernargPreload() ||
        F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
      continue;
    FunctionsToProcess.push_back(&F);
  }

  for (Function *FPtr : FunctionsToProcess) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(*FPtr);

    // Try to split byref struct arguments first. This may create a new
    // function and erase the old one.
    if (Function *NewF = trySplitKernelArguments(*FPtr, ST)) {
      FPtr = NewF;
      Changed = true;
    }

    Function &F = *FPtr;

    PreloadKernelArgInfo PreloadInfo(F, ST);
    uint64_t ExplicitArgOffset = 0;
    const DataLayout &DL = F.getDataLayout();
    const uint64_t BaseOffset = ST.getExplicitKernelArgOffset();
    unsigned NumPreloadsRequested = KernargPreloadCount;
    unsigned NumPreloadedExplicitArgs = 0;
    for (Argument &Arg : F.args()) {
      // Avoid incompatible attributes and guard against running this pass
      // twice. Note: byref struct arguments are handled by splitting them
      // into scalar arguments above via trySplitKernelArguments().
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

  Changed |= !FunctionsToErase.empty();
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
