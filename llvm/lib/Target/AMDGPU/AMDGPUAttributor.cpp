//===- AMDGPUAttributor.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass uses Attributor framework to deduce AMDGPU attributes.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Attributor.h"

#define DEBUG_TYPE "amdgpu-attributor"

using namespace llvm;

static cl::opt<unsigned> IndirectCallSpecializationThreshold(
    "amdgpu-indirect-call-specialization-threshold",
    cl::desc(
        "A threshold controls whether an indirect call will be specialized"),
    cl::init(3));

#define AMDGPU_ATTRIBUTE(Name, Str) Name##_POS,

enum ImplicitArgumentPositions {
#include "AMDGPUAttributes.def"
  LAST_ARG_POS
};

#define AMDGPU_ATTRIBUTE(Name, Str) Name = 1 << Name##_POS,

enum ImplicitArgumentMask {
  NOT_IMPLICIT_INPUT = 0,
#include "AMDGPUAttributes.def"
  ALL_ARGUMENT_MASK = (1 << LAST_ARG_POS) - 1
};

#define AMDGPU_ATTRIBUTE(Name, Str) {Name, Str},
static constexpr std::pair<ImplicitArgumentMask, StringLiteral>
    ImplicitAttrs[] = {
#include "AMDGPUAttributes.def"
};

// We do not need to note the x workitem or workgroup id because they are always
// initialized.
//
// TODO: We should not add the attributes if the known compile time workgroup
// size is 1 for y/z.
static ImplicitArgumentMask
intrinsicToAttrMask(Intrinsic::ID ID, bool &NonKernelOnly, bool &NeedsImplicit,
                    bool HasApertureRegs, bool SupportsGetDoorBellID,
                    unsigned CodeObjectVersion) {
  switch (ID) {
  case Intrinsic::amdgcn_workitem_id_x:
    NonKernelOnly = true;
    return WORKITEM_ID_X;
  case Intrinsic::amdgcn_workgroup_id_x:
    NonKernelOnly = true;
    return WORKGROUP_ID_X;
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::r600_read_tidig_y:
    return WORKITEM_ID_Y;
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::r600_read_tidig_z:
    return WORKITEM_ID_Z;
  case Intrinsic::amdgcn_workgroup_id_y:
  case Intrinsic::r600_read_tgid_y:
    return WORKGROUP_ID_Y;
  case Intrinsic::amdgcn_workgroup_id_z:
  case Intrinsic::r600_read_tgid_z:
    return WORKGROUP_ID_Z;
  case Intrinsic::amdgcn_lds_kernel_id:
    return LDS_KERNEL_ID;
  case Intrinsic::amdgcn_dispatch_ptr:
    return DISPATCH_PTR;
  case Intrinsic::amdgcn_dispatch_id:
    return DISPATCH_ID;
  case Intrinsic::amdgcn_implicitarg_ptr:
    return IMPLICIT_ARG_PTR;
  // Need queue_ptr anyway. But under V5, we also need implicitarg_ptr to access
  // queue_ptr.
  case Intrinsic::amdgcn_queue_ptr:
    NeedsImplicit = (CodeObjectVersion >= AMDGPU::AMDHSA_COV5);
    return QUEUE_PTR;
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private:
    if (HasApertureRegs)
      return NOT_IMPLICIT_INPUT;
    // Under V5, we need implicitarg_ptr + offsets to access private_base or
    // shared_base. For pre-V5, however, need to access them through queue_ptr +
    // offsets.
    return CodeObjectVersion >= AMDGPU::AMDHSA_COV5 ? IMPLICIT_ARG_PTR
                                                    : QUEUE_PTR;
  case Intrinsic::trap:
  case Intrinsic::debugtrap:
  case Intrinsic::ubsantrap:
    if (SupportsGetDoorBellID) // GetDoorbellID support implemented since V4.
      return CodeObjectVersion >= AMDGPU::AMDHSA_COV4 ? NOT_IMPLICIT_INPUT
                                                      : QUEUE_PTR;
    NeedsImplicit = (CodeObjectVersion >= AMDGPU::AMDHSA_COV5);
    return QUEUE_PTR;
  default:
    return NOT_IMPLICIT_INPUT;
  }
}

static bool castRequiresQueuePtr(unsigned SrcAS) {
  return SrcAS == AMDGPUAS::LOCAL_ADDRESS || SrcAS == AMDGPUAS::PRIVATE_ADDRESS;
}

static bool isDSAddress(const Constant *C) {
  const GlobalValue *GV = dyn_cast<GlobalValue>(C);
  if (!GV)
    return false;
  unsigned AS = GV->getAddressSpace();
  return AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS;
}

/// Returns true if the function requires the implicit argument be passed
/// regardless of the function contents.
static bool funcRequiresHostcallPtr(const Function &F) {
  // Sanitizers require the hostcall buffer passed in the implicit arguments.
  return F.hasFnAttribute(Attribute::SanitizeAddress) ||
         F.hasFnAttribute(Attribute::SanitizeThread) ||
         F.hasFnAttribute(Attribute::SanitizeMemory) ||
         F.hasFnAttribute(Attribute::SanitizeHWAddress) ||
         F.hasFnAttribute(Attribute::SanitizeMemTag);
}

namespace {
class AMDGPUInformationCache : public InformationCache {
public:
  AMDGPUInformationCache(const Module &M, AnalysisGetter &AG,
                         BumpPtrAllocator &Allocator,
                         SetVector<Function *> *CGSCC, TargetMachine &TM)
      : InformationCache(M, AG, Allocator, CGSCC), TM(TM),
        CodeObjectVersion(AMDGPU::getAMDHSACodeObjectVersion(M)) {}

  TargetMachine &TM;

  enum ConstantStatus : uint8_t {
    NONE = 0,
    DS_GLOBAL = 1 << 0,
    ADDR_SPACE_CAST_PRIVATE_TO_FLAT = 1 << 1,
    ADDR_SPACE_CAST_LOCAL_TO_FLAT = 1 << 2,
    ADDR_SPACE_CAST_BOTH_TO_FLAT =
        ADDR_SPACE_CAST_PRIVATE_TO_FLAT | ADDR_SPACE_CAST_LOCAL_TO_FLAT
  };

  /// Check if the subtarget has aperture regs.
  bool hasApertureRegs(Function &F) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.hasApertureRegs();
  }

  /// Check if the subtarget supports GetDoorbellID.
  bool supportsGetDoorbellID(Function &F) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.supportsGetDoorbellID();
  }

  std::optional<std::pair<unsigned, unsigned>>
  getFlatWorkGroupSizeAttr(const Function &F) const {
    auto R = AMDGPU::getIntegerPairAttribute(F, "amdgpu-flat-work-group-size");
    if (!R)
      return std::nullopt;
    return std::make_pair(R->first, *(R->second));
  }

  std::pair<unsigned, unsigned>
  getDefaultFlatWorkGroupSize(const Function &F) const {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.getDefaultFlatWorkGroupSize(F.getCallingConv());
  }

  std::pair<unsigned, unsigned>
  getMaximumFlatWorkGroupRange(const Function &F) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return {ST.getMinFlatWorkGroupSize(), ST.getMaxFlatWorkGroupSize()};
  }

  SmallVector<unsigned> getMaxNumWorkGroups(const Function &F) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.getMaxNumWorkGroups(F);
  }

  /// Get code object version.
  unsigned getCodeObjectVersion() const { return CodeObjectVersion; }

  /// Get the effective value of "amdgpu-waves-per-eu" for the function,
  /// accounting for the interaction with the passed value to use for
  /// "amdgpu-flat-work-group-size".
  std::pair<unsigned, unsigned>
  getWavesPerEU(const Function &F,
                std::pair<unsigned, unsigned> FlatWorkGroupSize) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.getWavesPerEU(FlatWorkGroupSize, getLDSSize(F), F);
  }

  std::optional<std::pair<unsigned, unsigned>>
  getWavesPerEUAttr(const Function &F) {
    auto Val = AMDGPU::getIntegerPairAttribute(F, "amdgpu-waves-per-eu",
                                               /*OnlyFirstRequired=*/true);
    if (!Val)
      return std::nullopt;
    if (!Val->second) {
      const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
      Val->second = ST.getMaxWavesPerEU();
    }
    return std::make_pair(Val->first, *(Val->second));
  }

  std::pair<unsigned, unsigned>
  getEffectiveWavesPerEU(const Function &F,
                         std::pair<unsigned, unsigned> WavesPerEU,
                         std::pair<unsigned, unsigned> FlatWorkGroupSize) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.getEffectiveWavesPerEU(WavesPerEU, FlatWorkGroupSize,
                                     getLDSSize(F));
  }

  unsigned getMaxWavesPerEU(const Function &F) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.getMaxWavesPerEU();
  }

  unsigned getMaxAddrSpace() const override {
    return AMDGPUAS::MAX_AMDGPU_ADDRESS;
  }

private:
  /// Check if the ConstantExpr \p CE uses an addrspacecast from private or
  /// local to flat. These casts may require the queue pointer.
  static uint8_t visitConstExpr(const ConstantExpr *CE) {
    uint8_t Status = NONE;

    if (CE->getOpcode() == Instruction::AddrSpaceCast) {
      unsigned SrcAS = CE->getOperand(0)->getType()->getPointerAddressSpace();
      if (SrcAS == AMDGPUAS::PRIVATE_ADDRESS)
        Status |= ADDR_SPACE_CAST_PRIVATE_TO_FLAT;
      else if (SrcAS == AMDGPUAS::LOCAL_ADDRESS)
        Status |= ADDR_SPACE_CAST_LOCAL_TO_FLAT;
    }

    return Status;
  }

  /// Returns the minimum amount of LDS space used by a workgroup running
  /// function \p F.
  static unsigned getLDSSize(const Function &F) {
    return AMDGPU::getIntegerPairAttribute(F, "amdgpu-lds-size",
                                           {0, UINT32_MAX}, true)
        .first;
  }

  /// Get the constant access bitmap for \p C.
  uint8_t getConstantAccess(const Constant *C,
                            SmallPtrSetImpl<const Constant *> &Visited) {
    auto It = ConstantStatus.find(C);
    if (It != ConstantStatus.end())
      return It->second;

    uint8_t Result = 0;
    if (isDSAddress(C))
      Result = DS_GLOBAL;

    if (const auto *CE = dyn_cast<ConstantExpr>(C))
      Result |= visitConstExpr(CE);

    for (const Use &U : C->operands()) {
      const auto *OpC = dyn_cast<Constant>(U);
      if (!OpC || !Visited.insert(OpC).second)
        continue;

      Result |= getConstantAccess(OpC, Visited);
    }
    return Result;
  }

public:
  /// Returns true if \p Fn needs the queue pointer because of \p C.
  bool needsQueuePtr(const Constant *C, Function &Fn) {
    bool IsNonEntryFunc = !AMDGPU::isEntryFunctionCC(Fn.getCallingConv());
    bool HasAperture = hasApertureRegs(Fn);

    // No need to explore the constants.
    if (!IsNonEntryFunc && HasAperture)
      return false;

    SmallPtrSet<const Constant *, 8> Visited;
    uint8_t Access = getConstantAccess(C, Visited);

    // We need to trap on DS globals in non-entry functions.
    if (IsNonEntryFunc && (Access & DS_GLOBAL))
      return true;

    return !HasAperture && (Access & ADDR_SPACE_CAST_BOTH_TO_FLAT);
  }

  bool checkConstForAddrSpaceCastFromPrivate(const Constant *C) {
    SmallPtrSet<const Constant *, 8> Visited;
    uint8_t Access = getConstantAccess(C, Visited);
    return Access & ADDR_SPACE_CAST_PRIVATE_TO_FLAT;
  }

private:
  /// Used to determine if the Constant needs the queue pointer.
  DenseMap<const Constant *, uint8_t> ConstantStatus;
  const unsigned CodeObjectVersion;
};

struct AAAMDAttributes
    : public StateWrapper<BitIntegerState<uint32_t, ALL_ARGUMENT_MASK, 0>,
                          AbstractAttribute> {
  using Base = StateWrapper<BitIntegerState<uint32_t, ALL_ARGUMENT_MASK, 0>,
                            AbstractAttribute>;

  AAAMDAttributes(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAAMDAttributes &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

  /// See AbstractAttribute::getName().
  StringRef getName() const override { return "AAAMDAttributes"; }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDAttributes.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};
const char AAAMDAttributes::ID = 0;

struct AAUniformWorkGroupSize
    : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAUniformWorkGroupSize(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAUniformWorkGroupSize &createForPosition(const IRPosition &IRP,
                                                   Attributor &A);

  /// See AbstractAttribute::getName().
  StringRef getName() const override { return "AAUniformWorkGroupSize"; }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDAttributes.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};
const char AAUniformWorkGroupSize::ID = 0;

struct AAUniformWorkGroupSizeFunction : public AAUniformWorkGroupSize {
  AAUniformWorkGroupSizeFunction(const IRPosition &IRP, Attributor &A)
      : AAUniformWorkGroupSize(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    CallingConv::ID CC = F->getCallingConv();

    if (CC != CallingConv::AMDGPU_KERNEL)
      return;

    bool InitialValue = false;
    if (F->hasFnAttribute("uniform-work-group-size"))
      InitialValue =
          F->getFnAttribute("uniform-work-group-size").getValueAsString() ==
          "true";

    if (InitialValue)
      indicateOptimisticFixpoint();
    else
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << "[AAUniformWorkGroupSize] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << "\n");

      const auto *CallerInfo = A.getAAFor<AAUniformWorkGroupSize>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerInfo || !CallerInfo->isValidState())
        return false;

      Change = Change | clampStateAndIndicateChange(this->getState(),
                                                    CallerInfo->getState());

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    AttrList.push_back(Attribute::get(Ctx, "uniform-work-group-size",
                                      getAssumed() ? "true" : "false"));
    return A.manifestAttrs(getIRPosition(), AttrList,
                           /* ForceReplace */ true);
  }

  bool isValidState() const override {
    // This state is always valid, even when the state is false.
    return true;
  }

  const std::string getAsStr(Attributor *) const override {
    return "AMDWorkGroupSize[" + std::to_string(getAssumed()) + "]";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

AAUniformWorkGroupSize &
AAUniformWorkGroupSize::createForPosition(const IRPosition &IRP,
                                          Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAUniformWorkGroupSizeFunction(IRP, A);
  llvm_unreachable(
      "AAUniformWorkGroupSize is only valid for function position");
}

struct AAAMDAttributesFunction : public AAAMDAttributes {
  AAAMDAttributesFunction(const IRPosition &IRP, Attributor &A)
      : AAAMDAttributes(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();

    // If the function requires the implicit arg pointer due to sanitizers,
    // assume it's needed even if explicitly marked as not requiring it.
    const bool NeedsHostcall = funcRequiresHostcallPtr(*F);
    if (NeedsHostcall) {
      removeAssumedBits(IMPLICIT_ARG_PTR);
      removeAssumedBits(HOSTCALL_PTR);
    }

    for (auto Attr : ImplicitAttrs) {
      if (NeedsHostcall &&
          (Attr.first == IMPLICIT_ARG_PTR || Attr.first == HOSTCALL_PTR))
        continue;

      if (F->hasFnAttribute(Attr.second))
        addKnownBits(Attr.first);
    }

    if (F->isDeclaration())
      return;

    // Ignore functions with graphics calling conventions, these are currently
    // not allowed to have kernel arguments.
    if (AMDGPU::isGraphics(F->getCallingConv())) {
      indicatePessimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAssociatedFunction();
    // The current assumed state used to determine a change.
    auto OrigAssumed = getAssumed();

    // Check for Intrinsics and propagate attributes.
    const AACallEdges *AAEdges = A.getAAFor<AACallEdges>(
        *this, this->getIRPosition(), DepClassTy::REQUIRED);
    if (!AAEdges || !AAEdges->isValidState() ||
        AAEdges->hasNonAsmUnknownCallee())
      return indicatePessimisticFixpoint();

    bool IsNonEntryFunc = !AMDGPU::isEntryFunctionCC(F->getCallingConv());

    bool NeedsImplicit = false;
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());
    bool HasApertureRegs = InfoCache.hasApertureRegs(*F);
    bool SupportsGetDoorbellID = InfoCache.supportsGetDoorbellID(*F);
    unsigned COV = InfoCache.getCodeObjectVersion();

    for (Function *Callee : AAEdges->getOptimisticEdges()) {
      Intrinsic::ID IID = Callee->getIntrinsicID();
      if (IID == Intrinsic::not_intrinsic) {
        const AAAMDAttributes *AAAMD = A.getAAFor<AAAMDAttributes>(
            *this, IRPosition::function(*Callee), DepClassTy::REQUIRED);
        if (!AAAMD || !AAAMD->isValidState())
          return indicatePessimisticFixpoint();
        *this &= *AAAMD;
        continue;
      }

      bool NonKernelOnly = false;
      ImplicitArgumentMask AttrMask =
          intrinsicToAttrMask(IID, NonKernelOnly, NeedsImplicit,
                              HasApertureRegs, SupportsGetDoorbellID, COV);
      if (AttrMask != NOT_IMPLICIT_INPUT) {
        if ((IsNonEntryFunc || !NonKernelOnly))
          removeAssumedBits(AttrMask);
      }
    }

    // Need implicitarg_ptr to acess queue_ptr, private_base, and shared_base.
    if (NeedsImplicit)
      removeAssumedBits(IMPLICIT_ARG_PTR);

    if (isAssumed(QUEUE_PTR) && checkForQueuePtr(A)) {
      // Under V5, we need implicitarg_ptr + offsets to access private_base or
      // shared_base. We do not actually need queue_ptr.
      if (COV >= 5)
        removeAssumedBits(IMPLICIT_ARG_PTR);
      else
        removeAssumedBits(QUEUE_PTR);
    }

    if (funcRetrievesMultigridSyncArg(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) &&
             "multigrid_sync_arg needs implicitarg_ptr");
      removeAssumedBits(MULTIGRID_SYNC_ARG);
    }

    if (funcRetrievesHostcallPtr(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) && "hostcall needs implicitarg_ptr");
      removeAssumedBits(HOSTCALL_PTR);
    }

    if (funcRetrievesHeapPtr(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) && "heap_ptr needs implicitarg_ptr");
      removeAssumedBits(HEAP_PTR);
    }

    if (isAssumed(QUEUE_PTR) && funcRetrievesQueuePtr(A, COV)) {
      assert(!isAssumed(IMPLICIT_ARG_PTR) && "queue_ptr needs implicitarg_ptr");
      removeAssumedBits(QUEUE_PTR);
    }

    if (isAssumed(LDS_KERNEL_ID) && funcRetrievesLDSKernelId(A)) {
      removeAssumedBits(LDS_KERNEL_ID);
    }

    if (isAssumed(DEFAULT_QUEUE) && funcRetrievesDefaultQueue(A, COV))
      removeAssumedBits(DEFAULT_QUEUE);

    if (isAssumed(COMPLETION_ACTION) && funcRetrievesCompletionAction(A, COV))
      removeAssumedBits(COMPLETION_ACTION);

    if (isAssumed(FLAT_SCRATCH_INIT) && needFlatScratchInit(A))
      removeAssumedBits(FLAT_SCRATCH_INIT);

    return getAssumed() != OrigAssumed ? ChangeStatus::CHANGED
                                       : ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    for (auto Attr : ImplicitAttrs) {
      if (isKnown(Attr.first))
        AttrList.push_back(Attribute::get(Ctx, Attr.second));
    }

    return A.manifestAttrs(getIRPosition(), AttrList,
                           /* ForceReplace */ true);
  }

  const std::string getAsStr(Attributor *) const override {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << "AMDInfo[";
    for (auto Attr : ImplicitAttrs)
      if (isAssumed(Attr.first))
        OS << ' ' << Attr.second;
    OS << " ]";
    return OS.str();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

private:
  bool checkForQueuePtr(Attributor &A) {
    Function *F = getAssociatedFunction();
    bool IsNonEntryFunc = !AMDGPU::isEntryFunctionCC(F->getCallingConv());

    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());

    bool NeedsQueuePtr = false;

    auto CheckAddrSpaceCasts = [&](Instruction &I) {
      unsigned SrcAS = static_cast<AddrSpaceCastInst &>(I).getSrcAddressSpace();
      if (castRequiresQueuePtr(SrcAS)) {
        NeedsQueuePtr = true;
        return false;
      }
      return true;
    };

    bool HasApertureRegs = InfoCache.hasApertureRegs(*F);

    // `checkForAllInstructions` is much more cheaper than going through all
    // instructions, try it first.

    // The queue pointer is not needed if aperture regs is present.
    if (!HasApertureRegs) {
      bool UsedAssumedInformation = false;
      A.checkForAllInstructions(CheckAddrSpaceCasts, *this,
                                {Instruction::AddrSpaceCast},
                                UsedAssumedInformation);
    }

    // If we found  that we need the queue pointer, nothing else to do.
    if (NeedsQueuePtr)
      return true;

    if (!IsNonEntryFunc && HasApertureRegs)
      return false;

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        for (const Use &U : I.operands()) {
          if (const auto *C = dyn_cast<Constant>(U)) {
            if (InfoCache.needsQueuePtr(C, *F))
              return true;
          }
        }
      }
    }

    return false;
  }

  bool funcRetrievesMultigridSyncArg(Attributor &A, unsigned COV) {
    auto Pos = llvm::AMDGPU::getMultigridSyncArgImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesHostcallPtr(Attributor &A, unsigned COV) {
    auto Pos = llvm::AMDGPU::getHostcallImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesDefaultQueue(Attributor &A, unsigned COV) {
    auto Pos = llvm::AMDGPU::getDefaultQueueImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesCompletionAction(Attributor &A, unsigned COV) {
    auto Pos = llvm::AMDGPU::getCompletionActionImplicitArgPosition(COV);
    AA::RangeTy Range(Pos, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesHeapPtr(Attributor &A, unsigned COV) {
    if (COV < 5)
      return false;
    AA::RangeTy Range(AMDGPU::ImplicitArg::HEAP_PTR_OFFSET, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesQueuePtr(Attributor &A, unsigned COV) {
    if (COV < 5)
      return false;
    AA::RangeTy Range(AMDGPU::ImplicitArg::QUEUE_PTR_OFFSET, 8);
    return funcRetrievesImplicitKernelArg(A, Range);
  }

  bool funcRetrievesImplicitKernelArg(Attributor &A, AA::RangeTy Range) {
    // Check if this is a call to the implicitarg_ptr builtin and it
    // is used to retrieve the hostcall pointer. The implicit arg for
    // hostcall is not used only if every use of the implicitarg_ptr
    // is a load that clearly does not retrieve any byte of the
    // hostcall pointer. We check this by tracing all the uses of the
    // initial call to the implicitarg_ptr intrinsic.
    auto DoesNotLeadToKernelArgLoc = [&](Instruction &I) {
      auto &Call = cast<CallBase>(I);
      if (Call.getIntrinsicID() != Intrinsic::amdgcn_implicitarg_ptr)
        return true;

      const auto *PointerInfoAA = A.getAAFor<AAPointerInfo>(
          *this, IRPosition::callsite_returned(Call), DepClassTy::REQUIRED);
      if (!PointerInfoAA || !PointerInfoAA->getState().isValidState())
        return false;

      return PointerInfoAA->forallInterferingAccesses(
          Range, [](const AAPointerInfo::Access &Acc, bool IsExact) {
            return Acc.getRemoteInst()->isDroppable();
          });
    };

    bool UsedAssumedInformation = false;
    return !A.checkForAllCallLikeInstructions(DoesNotLeadToKernelArgLoc, *this,
                                              UsedAssumedInformation);
  }

  bool funcRetrievesLDSKernelId(Attributor &A) {
    auto DoesNotRetrieve = [&](Instruction &I) {
      auto &Call = cast<CallBase>(I);
      return Call.getIntrinsicID() != Intrinsic::amdgcn_lds_kernel_id;
    };
    bool UsedAssumedInformation = false;
    return !A.checkForAllCallLikeInstructions(DoesNotRetrieve, *this,
                                              UsedAssumedInformation);
  }

  // Returns true if FlatScratchInit is needed, i.e., no-flat-scratch-init is
  // not to be set.
  bool needFlatScratchInit(Attributor &A) {
    assert(isAssumed(FLAT_SCRATCH_INIT)); // only called if the bit is still set

    // Check all AddrSpaceCast instructions. FlatScratchInit is needed if
    // there is a cast from PRIVATE_ADDRESS.
    auto AddrSpaceCastNotFromPrivate = [](Instruction &I) {
      return cast<AddrSpaceCastInst>(I).getSrcAddressSpace() !=
             AMDGPUAS::PRIVATE_ADDRESS;
    };

    bool UsedAssumedInformation = false;
    if (!A.checkForAllInstructions(AddrSpaceCastNotFromPrivate, *this,
                                   {Instruction::AddrSpaceCast},
                                   UsedAssumedInformation))
      return true;

    // Check for addrSpaceCast from PRIVATE_ADDRESS in constant expressions
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());

    Function *F = getAssociatedFunction();
    for (Instruction &I : instructions(F)) {
      for (const Use &U : I.operands()) {
        if (const auto *C = dyn_cast<Constant>(U)) {
          if (InfoCache.checkConstForAddrSpaceCastFromPrivate(C))
            return true;
        }
      }
    }

    // Finally check callees.

    // This is called on each callee; false means callee shouldn't have
    // no-flat-scratch-init.
    auto CheckForNoFlatScratchInit = [&](Instruction &I) {
      const auto &CB = cast<CallBase>(I);
      const Function *Callee = CB.getCalledFunction();

      // Callee == 0 for inline asm or indirect call with known callees.
      // In the latter case, updateImpl() already checked the callees and we
      // know their FLAT_SCRATCH_INIT bit is set.
      // If function has indirect call with unknown callees, the bit is
      // already removed in updateImpl() and execution won't reach here.
      if (!Callee)
        return true;

      return Callee->getIntrinsicID() !=
             Intrinsic::amdgcn_addrspacecast_nonnull;
    };

    UsedAssumedInformation = false;
    // If any callee is false (i.e. need FlatScratchInit),
    // checkForAllCallLikeInstructions returns false, in which case this
    // function returns true.
    return !A.checkForAllCallLikeInstructions(CheckForNoFlatScratchInit, *this,
                                              UsedAssumedInformation);
  }
};

AAAMDAttributes &AAAMDAttributes::createForPosition(const IRPosition &IRP,
                                                    Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAAMDAttributesFunction(IRP, A);
  llvm_unreachable("AAAMDAttributes is only valid for function position");
}

/// Base class to derive different size ranges.
struct AAAMDSizeRangeAttribute
    : public StateWrapper<IntegerRangeState, AbstractAttribute, uint32_t> {
  using Base = StateWrapper<IntegerRangeState, AbstractAttribute, uint32_t>;

  StringRef AttrName;

  AAAMDSizeRangeAttribute(const IRPosition &IRP, Attributor &A,
                          StringRef AttrName)
      : Base(IRP, 32), AttrName(AttrName) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

  template <class AttributeImpl> ChangeStatus updateImplImpl(Attributor &A) {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << '[' << getName() << "] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << '\n');

      const auto *CallerInfo = A.getAAFor<AttributeImpl>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerInfo || !CallerInfo->isValidState())
        return false;

      Change |=
          clampStateAndIndicateChange(this->getState(), CallerInfo->getState());

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this,
                                /*RequireAllCallSites=*/true,
                                AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  /// Clamp the assumed range to the default value ([Min, Max]) and emit the
  /// attribute if it is not same as default.
  ChangeStatus
  emitAttributeIfNotDefaultAfterClamp(Attributor &A,
                                      std::pair<unsigned, unsigned> Default) {
    auto [Min, Max] = Default;
    unsigned Lower = getAssumed().getLower().getZExtValue();
    unsigned Upper = getAssumed().getUpper().getZExtValue();

    // Clamp the range to the default value.
    if (Lower < Min)
      Lower = Min;
    if (Upper > Max + 1)
      Upper = Max + 1;

    // No manifest if the value is invalid or same as default after clamp.
    if ((Lower == Min && Upper == Max + 1) || (Upper < Lower))
      return ChangeStatus::UNCHANGED;

    Function *F = getAssociatedFunction();
    LLVMContext &Ctx = F->getContext();
    SmallString<10> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << Lower << ',' << Upper - 1;
    return A.manifestAttrs(getIRPosition(),
                           {Attribute::get(Ctx, AttrName, OS.str())},
                           /*ForceReplace=*/true);
  }

  const std::string getAsStr(Attributor *) const override {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << getName() << '[';
    OS << getAssumed().getLower() << ',' << getAssumed().getUpper() - 1;
    OS << ']';
    return OS.str();
  }
};

/// Propagate amdgpu-flat-work-group-size attribute.
struct AAAMDFlatWorkGroupSize : public AAAMDSizeRangeAttribute {
  AAAMDFlatWorkGroupSize(const IRPosition &IRP, Attributor &A)
      : AAAMDSizeRangeAttribute(IRP, A, "amdgpu-flat-work-group-size") {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());

    bool HasAttr = false;
    auto Range = InfoCache.getDefaultFlatWorkGroupSize(*F);
    auto MaxRange = InfoCache.getMaximumFlatWorkGroupRange(*F);

    if (auto Attr = InfoCache.getFlatWorkGroupSizeAttr(*F)) {
      // We only consider an attribute that is not max range because the front
      // end always emits the attribute, unfortunately, and sometimes it emits
      // the max range.
      if (*Attr != MaxRange) {
        Range = *Attr;
        HasAttr = true;
      }
    }

    // We don't want to directly clamp the state if it's the max range because
    // that is basically the worst state.
    if (Range == MaxRange)
      return;

    auto [Min, Max] = Range;
    ConstantRange CR(APInt(32, Min), APInt(32, Max + 1));
    IntegerRangeState IRS(CR);
    clampStateAndIndicateChange(this->getState(), IRS);

    if (HasAttr || AMDGPU::isEntryFunctionCC(F->getCallingConv()))
      indicateOptimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    return updateImplImpl<AAAMDFlatWorkGroupSize>(A);
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AAAMDFlatWorkGroupSize &createForPosition(const IRPosition &IRP,
                                                   Attributor &A);

  ChangeStatus manifest(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());
    return emitAttributeIfNotDefaultAfterClamp(
        A, InfoCache.getMaximumFlatWorkGroupRange(*F));
  }

  /// See AbstractAttribute::getName()
  StringRef getName() const override { return "AAAMDFlatWorkGroupSize"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDFlatWorkGroupSize
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

const char AAAMDFlatWorkGroupSize::ID = 0;

AAAMDFlatWorkGroupSize &
AAAMDFlatWorkGroupSize::createForPosition(const IRPosition &IRP,
                                          Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAAMDFlatWorkGroupSize(IRP, A);
  llvm_unreachable(
      "AAAMDFlatWorkGroupSize is only valid for function position");
}

struct TupleDecIntegerRangeState : public AbstractState {
  DecIntegerState<uint32_t> X, Y, Z;

  bool isValidState() const override {
    return X.isValidState() && Y.isValidState() && Z.isValidState();
  }

  bool isAtFixpoint() const override {
    return X.isAtFixpoint() && Y.isAtFixpoint() && Z.isAtFixpoint();
  }

  ChangeStatus indicateOptimisticFixpoint() override {
    return X.indicateOptimisticFixpoint() | Y.indicateOptimisticFixpoint() |
           Z.indicateOptimisticFixpoint();
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    return X.indicatePessimisticFixpoint() | Y.indicatePessimisticFixpoint() |
           Z.indicatePessimisticFixpoint();
  }

  TupleDecIntegerRangeState operator^=(const TupleDecIntegerRangeState &Other) {
    X ^= Other.X;
    Y ^= Other.Y;
    Z ^= Other.Z;
    return *this;
  }

  bool operator==(const TupleDecIntegerRangeState &Other) const {
    return X == Other.X && Y == Other.Y && Z == Other.Z;
  }

  TupleDecIntegerRangeState &getAssumed() { return *this; }
  const TupleDecIntegerRangeState &getAssumed() const { return *this; }
};

using AAAMDMaxNumWorkgroupsState =
    StateWrapper<TupleDecIntegerRangeState, AbstractAttribute, uint32_t>;

/// Propagate amdgpu-max-num-workgroups attribute.
struct AAAMDMaxNumWorkgroups
    : public StateWrapper<TupleDecIntegerRangeState, AbstractAttribute> {
  using Base = StateWrapper<TupleDecIntegerRangeState, AbstractAttribute>;

  AAAMDMaxNumWorkgroups(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());

    SmallVector<unsigned> MaxNumWorkgroups = InfoCache.getMaxNumWorkGroups(*F);

    X.takeKnownMinimum(MaxNumWorkgroups[0]);
    Y.takeKnownMinimum(MaxNumWorkgroups[1]);
    Z.takeKnownMinimum(MaxNumWorkgroups[2]);

    if (AMDGPU::isEntryFunctionCC(F->getCallingConv()))
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << "[AAAMDMaxNumWorkgroups] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << '\n');

      const auto *CallerInfo = A.getAAFor<AAAMDMaxNumWorkgroups>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerInfo || !CallerInfo->isValidState())
        return false;

      Change |=
          clampStateAndIndicateChange(this->getState(), CallerInfo->getState());
      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this,
                                /*RequireAllCallSites=*/true,
                                AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AAAMDMaxNumWorkgroups &createForPosition(const IRPosition &IRP,
                                                  Attributor &A);

  ChangeStatus manifest(Attributor &A) override {
    Function *F = getAssociatedFunction();
    LLVMContext &Ctx = F->getContext();
    SmallString<32> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << X.getAssumed() << ',' << Y.getAssumed() << ',' << Z.getAssumed();

    // TODO: Should annotate loads of the group size for this to do anything
    // useful.
    return A.manifestAttrs(
        getIRPosition(),
        {Attribute::get(Ctx, "amdgpu-max-num-workgroups", OS.str())},
        /* ForceReplace= */ true);
  }

  StringRef getName() const override { return "AAAMDMaxNumWorkgroups"; }

  const std::string getAsStr(Attributor *) const override {
    std::string Buffer = "AAAMDMaxNumWorkgroupsState[";
    raw_string_ostream OS(Buffer);
    OS << X.getAssumed() << ',' << Y.getAssumed() << ',' << Z.getAssumed()
       << ']';
    return OS.str();
  }

  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDMaxNumWorkgroups
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  void trackStatistics() const override {}

  /// Unique ID (due to the unique address)
  static const char ID;
};

const char AAAMDMaxNumWorkgroups::ID = 0;

AAAMDMaxNumWorkgroups &
AAAMDMaxNumWorkgroups::createForPosition(const IRPosition &IRP, Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAAMDMaxNumWorkgroups(IRP, A);
  llvm_unreachable("AAAMDMaxNumWorkgroups is only valid for function position");
}

/// Propagate amdgpu-waves-per-eu attribute.
struct AAAMDWavesPerEU : public AAAMDSizeRangeAttribute {
  AAAMDWavesPerEU(const IRPosition &IRP, Attributor &A)
      : AAAMDSizeRangeAttribute(IRP, A, "amdgpu-waves-per-eu") {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());

    // If the attribute exists, we will honor it if it is not the default.
    if (auto Attr = InfoCache.getWavesPerEUAttr(*F)) {
      std::pair<unsigned, unsigned> MaxWavesPerEURange{
          1U, InfoCache.getMaxWavesPerEU(*F)};
      if (*Attr != MaxWavesPerEURange) {
        auto [Min, Max] = *Attr;
        ConstantRange Range(APInt(32, Min), APInt(32, Max + 1));
        IntegerRangeState RangeState(Range);
        this->getState() = RangeState;
        indicateOptimisticFixpoint();
        return;
      }
    }

    if (AMDGPU::isEntryFunctionCC(F->getCallingConv()))
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      Function *Func = getAssociatedFunction();
      LLVM_DEBUG(dbgs() << '[' << getName() << "] Call " << Caller->getName()
                        << "->" << Func->getName() << '\n');
      (void)Func;

      const auto *CallerAA = A.getAAFor<AAAMDWavesPerEU>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerAA || !CallerAA->isValidState())
        return false;

      ConstantRange Assumed = getAssumed();
      unsigned Min = std::max(Assumed.getLower().getZExtValue(),
                              CallerAA->getAssumed().getLower().getZExtValue());
      unsigned Max = std::max(Assumed.getUpper().getZExtValue(),
                              CallerAA->getAssumed().getUpper().getZExtValue());
      ConstantRange Range(APInt(32, Min), APInt(32, Max));
      IntegerRangeState RangeState(Range);
      getState() = RangeState;
      Change |= getState() == Assumed ? ChangeStatus::UNCHANGED
                                      : ChangeStatus::CHANGED;

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  /// Create an abstract attribute view for the position \p IRP.
  static AAAMDWavesPerEU &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

  ChangeStatus manifest(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());
    return emitAttributeIfNotDefaultAfterClamp(
        A, {1U, InfoCache.getMaxWavesPerEU(*F)});
  }

  /// See AbstractAttribute::getName()
  StringRef getName() const override { return "AAAMDWavesPerEU"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDWavesPerEU
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

const char AAAMDWavesPerEU::ID = 0;

AAAMDWavesPerEU &AAAMDWavesPerEU::createForPosition(const IRPosition &IRP,
                                                    Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAAMDWavesPerEU(IRP, A);
  llvm_unreachable("AAAMDWavesPerEU is only valid for function position");
}

static bool inlineAsmUsesAGPRs(const InlineAsm *IA) {
  for (const auto &CI : IA->ParseConstraints()) {
    for (StringRef Code : CI.Codes) {
      Code.consume_front("{");
      if (Code.starts_with("a"))
        return true;
    }
  }

  return false;
}

// TODO: Migrate to range merge of amdgpu-agpr-alloc.
// FIXME: Why is this using Attribute::NoUnwind?
struct AAAMDGPUNoAGPR
    : public IRAttribute<Attribute::NoUnwind,
                         StateWrapper<BooleanState, AbstractAttribute>,
                         AAAMDGPUNoAGPR> {
  AAAMDGPUNoAGPR(const IRPosition &IRP, Attributor &A) : IRAttribute(IRP) {}

  static AAAMDGPUNoAGPR &createForPosition(const IRPosition &IRP,
                                           Attributor &A) {
    if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
      return *new (A.Allocator) AAAMDGPUNoAGPR(IRP, A);
    llvm_unreachable("AAAMDGPUNoAGPR is only valid for function position");
  }

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    auto [MinNumAGPR, MaxNumAGPR] =
        AMDGPU::getIntegerPairAttribute(*F, "amdgpu-agpr-alloc", {~0u, ~0u},
                                        /*OnlyFirstRequired=*/true);
    if (MinNumAGPR == 0)
      indicateOptimisticFixpoint();
  }

  const std::string getAsStr(Attributor *A) const override {
    return getAssumed() ? "amdgpu-no-agpr" : "amdgpu-maybe-agpr";
  }

  void trackStatistics() const override {}

  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Use AACallEdges, but then we need a way to inspect asm edges.

    auto CheckForNoAGPRs = [&](Instruction &I) {
      const auto &CB = cast<CallBase>(I);
      const Value *CalleeOp = CB.getCalledOperand();
      const Function *Callee = dyn_cast<Function>(CalleeOp);
      if (!Callee) {
        if (const InlineAsm *IA = dyn_cast<InlineAsm>(CalleeOp))
          return !inlineAsmUsesAGPRs(IA);
        return false;
      }

      // Some intrinsics may use AGPRs, but if we have a choice, we are not
      // required to use AGPRs.
      if (Callee->isIntrinsic())
        return true;

      // TODO: Handle callsite attributes
      const auto *CalleeInfo = A.getAAFor<AAAMDGPUNoAGPR>(
          *this, IRPosition::function(*Callee), DepClassTy::REQUIRED);
      return CalleeInfo && CalleeInfo->isValidState() &&
             CalleeInfo->getAssumed();
    };

    bool UsedAssumedInformation = false;
    if (!A.checkForAllCallLikeInstructions(CheckForNoAGPRs, *this,
                                           UsedAssumedInformation))
      return indicatePessimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    if (!getAssumed())
      return ChangeStatus::UNCHANGED;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();
    return A.manifestAttrs(getIRPosition(),
                           {Attribute::get(Ctx, "amdgpu-agpr-alloc", "0")});
  }

  StringRef getName() const override { return "AAAMDGPUNoAGPR"; }
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDGPUNoAGPRs
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  static const char ID;
};

const char AAAMDGPUNoAGPR::ID = 0;

/// Performs the final check and updates the 'amdgpu-waves-per-eu' attribute
/// based on the finalized 'amdgpu-flat-work-group-size' attribute.
/// Both attributes start with narrow ranges that expand during iteration.
/// However, a narrower flat-workgroup-size leads to a wider waves-per-eu range,
/// preventing optimal updates later. Therefore, waves-per-eu can't be updated
/// with intermediate values during the attributor run. We defer the
/// finalization of waves-per-eu until after the flat-workgroup-size is
/// finalized.
/// TODO: Remove this and move similar logic back into the attributor run once
/// we have a better representation for waves-per-eu.
static bool updateWavesPerEU(Module &M, TargetMachine &TM) {
  bool Changed = false;

  LLVMContext &Ctx = M.getContext();

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);

    std::optional<std::pair<unsigned, std::optional<unsigned>>>
        FlatWgrpSizeAttr =
            AMDGPU::getIntegerPairAttribute(F, "amdgpu-flat-work-group-size");

    unsigned MinWavesPerEU = ST.getMinWavesPerEU();
    unsigned MaxWavesPerEU = ST.getMaxWavesPerEU();

    unsigned MinFlatWgrpSize = ST.getMinFlatWorkGroupSize();
    unsigned MaxFlatWgrpSize = ST.getMaxFlatWorkGroupSize();
    if (FlatWgrpSizeAttr.has_value()) {
      MinFlatWgrpSize = FlatWgrpSizeAttr->first;
      MaxFlatWgrpSize = *(FlatWgrpSizeAttr->second);
    }

    // Start with the "best" range.
    unsigned Min = MinWavesPerEU;
    unsigned Max = MinWavesPerEU;

    // Compute the range from flat workgroup size. `getWavesPerEU` will also
    // account for the 'amdgpu-waves-er-eu' attribute.
    auto [MinFromFlatWgrpSize, MaxFromFlatWgrpSize] =
        ST.getWavesPerEU(F, {MinFlatWgrpSize, MaxFlatWgrpSize});

    // For the lower bound, we have to "tighten" it.
    Min = std::max(Min, MinFromFlatWgrpSize);
    // For the upper bound, we have to "extend" it.
    Max = std::max(Max, MaxFromFlatWgrpSize);

    // Clamp the range to the max range.
    Min = std::max(Min, MinWavesPerEU);
    Max = std::min(Max, MaxWavesPerEU);

    // Update the attribute if it is not the max.
    if (Min != MinWavesPerEU || Max != MaxWavesPerEU) {
      SmallString<10> Buffer;
      raw_svector_ostream OS(Buffer);
      OS << Min << ',' << Max;
      Attribute OldAttr = F.getFnAttribute("amdgpu-waves-per-eu");
      Attribute NewAttr = Attribute::get(Ctx, "amdgpu-waves-per-eu", OS.str());
      F.addFnAttr(NewAttr);
      Changed |= OldAttr == NewAttr;
    }
  }

  return Changed;
}

static bool runImpl(Module &M, AnalysisGetter &AG, TargetMachine &TM,
                    AMDGPUAttributorOptions Options,
                    ThinOrFullLTOPhase LTOPhase) {
  SetVector<Function *> Functions;
  for (Function &F : M) {
    if (!F.isIntrinsic())
      Functions.insert(&F);
  }

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  AMDGPUInformationCache InfoCache(M, AG, Allocator, nullptr, TM);
  DenseSet<const char *> Allowed(
      {&AAAMDAttributes::ID, &AAUniformWorkGroupSize::ID,
       &AAPotentialValues::ID, &AAAMDFlatWorkGroupSize::ID,
       &AAAMDMaxNumWorkgroups::ID, &AAAMDWavesPerEU::ID, &AAAMDGPUNoAGPR::ID,
       &AACallEdges::ID, &AAPointerInfo::ID, &AAPotentialConstantValues::ID,
       &AAUnderlyingObjects::ID, &AANoAliasAddrSpace::ID, &AAAddressSpace::ID,
       &AAIndirectCallInfo::ID});

  AttributorConfig AC(CGUpdater);
  AC.IsClosedWorldModule = Options.IsClosedWorld;
  AC.Allowed = &Allowed;
  AC.IsModulePass = true;
  AC.DefaultInitializeLiveInternals = false;
  AC.IndirectCalleeSpecializationCallback =
      [](Attributor &A, const AbstractAttribute &AA, CallBase &CB,
         Function &Callee, unsigned NumAssumedCallees) {
        return !AMDGPU::isEntryFunctionCC(Callee.getCallingConv()) &&
               (NumAssumedCallees <= IndirectCallSpecializationThreshold);
      };
  AC.IPOAmendableCB = [](const Function &F) {
    return F.getCallingConv() == CallingConv::AMDGPU_KERNEL;
  };

  Attributor A(Functions, InfoCache, AC);

  LLVM_DEBUG({
    StringRef LTOPhaseStr = to_string(LTOPhase);
    dbgs() << "[AMDGPUAttributor] Running at phase " << LTOPhaseStr << '\n'
           << "[AMDGPUAttributor] Module " << M.getName() << " is "
           << (AC.IsClosedWorldModule ? "" : "not ")
           << "assumed to be a closed world.\n";
  });

  for (auto *F : Functions) {
    A.getOrCreateAAFor<AAAMDAttributes>(IRPosition::function(*F));
    A.getOrCreateAAFor<AAUniformWorkGroupSize>(IRPosition::function(*F));
    A.getOrCreateAAFor<AAAMDMaxNumWorkgroups>(IRPosition::function(*F));
    A.getOrCreateAAFor<AAAMDGPUNoAGPR>(IRPosition::function(*F));
    CallingConv::ID CC = F->getCallingConv();
    if (!AMDGPU::isEntryFunctionCC(CC)) {
      A.getOrCreateAAFor<AAAMDFlatWorkGroupSize>(IRPosition::function(*F));
      A.getOrCreateAAFor<AAAMDWavesPerEU>(IRPosition::function(*F));
    }

    for (auto &I : instructions(F)) {
      Value *Ptr = nullptr;
      if (auto *LI = dyn_cast<LoadInst>(&I))
        Ptr = LI->getPointerOperand();
      else if (auto *SI = dyn_cast<StoreInst>(&I))
        Ptr = SI->getPointerOperand();
      else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
        Ptr = RMW->getPointerOperand();
      else if (auto *CmpX = dyn_cast<AtomicCmpXchgInst>(&I))
        Ptr = CmpX->getPointerOperand();

      if (Ptr) {
        A.getOrCreateAAFor<AAAddressSpace>(IRPosition::value(*Ptr));
        A.getOrCreateAAFor<AANoAliasAddrSpace>(IRPosition::value(*Ptr));
      }
    }
  }

  bool Changed = A.run() == ChangeStatus::CHANGED;

  Changed |= updateWavesPerEU(M, TM);

  return Changed;
}
} // namespace

PreservedAnalyses llvm::AMDGPUAttributorPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  AnalysisGetter AG(FAM);

  // TODO: Probably preserves CFG
  return runImpl(M, AG, TM, Options, LTOPhase) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}
