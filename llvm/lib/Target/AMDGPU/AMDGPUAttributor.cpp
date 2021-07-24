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
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Attributor.h"

#define DEBUG_TYPE "amdgpu-attributor"

using namespace llvm;

static constexpr StringLiteral ImplicitAttrNames[] = {
    // X ids unnecessarily propagated to kernels.
    "amdgpu-work-item-id-x",  "amdgpu-work-item-id-y",
    "amdgpu-work-item-id-z",  "amdgpu-work-group-id-x",
    "amdgpu-work-group-id-y", "amdgpu-work-group-id-z",
    "amdgpu-dispatch-ptr",    "amdgpu-dispatch-id",
    "amdgpu-queue-ptr",       "amdgpu-implicitarg-ptr"};

// We do not need to note the x workitem or workgroup id because they are always
// initialized.
//
// TODO: We should not add the attributes if the known compile time workgroup
// size is 1 for y/z.
static StringRef intrinsicToAttrName(Intrinsic::ID ID, bool &NonKernelOnly,
                                     bool &IsQueuePtr) {
  switch (ID) {
  case Intrinsic::amdgcn_workitem_id_x:
    NonKernelOnly = true;
    return "amdgpu-work-item-id-x";
  case Intrinsic::amdgcn_workgroup_id_x:
    NonKernelOnly = true;
    return "amdgpu-work-group-id-x";
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::r600_read_tidig_y:
    return "amdgpu-work-item-id-y";
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::r600_read_tidig_z:
    return "amdgpu-work-item-id-z";
  case Intrinsic::amdgcn_workgroup_id_y:
  case Intrinsic::r600_read_tgid_y:
    return "amdgpu-work-group-id-y";
  case Intrinsic::amdgcn_workgroup_id_z:
  case Intrinsic::r600_read_tgid_z:
    return "amdgpu-work-group-id-z";
  case Intrinsic::amdgcn_dispatch_ptr:
    return "amdgpu-dispatch-ptr";
  case Intrinsic::amdgcn_dispatch_id:
    return "amdgpu-dispatch-id";
  case Intrinsic::amdgcn_kernarg_segment_ptr:
    return "amdgpu-kernarg-segment-ptr";
  case Intrinsic::amdgcn_implicitarg_ptr:
    return "amdgpu-implicitarg-ptr";
  case Intrinsic::amdgcn_queue_ptr:
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private:
    // TODO: Does not require queue ptr on gfx9+
  case Intrinsic::trap:
  case Intrinsic::debugtrap:
    IsQueuePtr = true;
    return "amdgpu-queue-ptr";
  default:
    return "";
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

class AMDGPUInformationCache : public InformationCache {
public:
  AMDGPUInformationCache(const Module &M, AnalysisGetter &AG,
                         BumpPtrAllocator &Allocator,
                         SetVector<Function *> *CGSCC, TargetMachine &TM)
      : InformationCache(M, AG, Allocator, CGSCC), TM(TM) {}
  TargetMachine &TM;

  enum ConstantStatus { DS_GLOBAL = 1 << 0, ADDR_SPACE_CAST = 1 << 1 };

  /// Check if the subtarget has aperture regs.
  bool hasApertureRegs(Function &F) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    return ST.hasApertureRegs();
  }

private:
  /// Check if the ConstantExpr \p CE requires queue ptr attribute.
  static bool visitConstExpr(const ConstantExpr *CE) {
    if (CE->getOpcode() == Instruction::AddrSpaceCast) {
      unsigned SrcAS = CE->getOperand(0)->getType()->getPointerAddressSpace();
      return castRequiresQueuePtr(SrcAS);
    }
    return false;
  }

  /// Get the constant access bitmap for \p C.
  uint8_t getConstantAccess(const Constant *C) {
    auto It = ConstantStatus.find(C);
    if (It != ConstantStatus.end())
      return It->second;

    uint8_t Result = 0;
    if (isDSAddress(C))
      Result = DS_GLOBAL;

    if (const auto *CE = dyn_cast<ConstantExpr>(C))
      if (visitConstExpr(CE))
        Result |= ADDR_SPACE_CAST;

    for (const Use &U : C->operands()) {
      const auto *OpC = dyn_cast<Constant>(U);
      if (!OpC)
        continue;

      Result |= getConstantAccess(OpC);
    }
    return Result;
  }

public:
  /// Returns true if \p Fn needs a queue ptr attribute because of \p C.
  bool needsQueuePtr(const Constant *C, Function &Fn) {
    bool IsNonEntryFunc = !AMDGPU::isEntryFunctionCC(Fn.getCallingConv());
    bool HasAperture = hasApertureRegs(Fn);

    // No need to explore the constants.
    if (!IsNonEntryFunc && HasAperture)
      return false;

    uint8_t Access = getConstantAccess(C);

    // We need to trap on DS globals in non-entry functions.
    if (IsNonEntryFunc && (Access & DS_GLOBAL))
      return true;

    return !HasAperture && (Access & ADDR_SPACE_CAST);
  }

private:
  /// Used to determine if the Constant needs a queue ptr attribute.
  DenseMap<const Constant *, uint8_t> ConstantStatus;
};

struct AAAMDAttributes : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAAMDAttributes(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAAMDAttributes &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

  /// See AbstractAttribute::getName().
  const std::string getName() const override { return "AAAMDAttributes"; }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAAMDAttributes.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  virtual const DenseSet<StringRef> &getAttributes() const = 0;

  /// Unique ID (due to the unique address)
  static const char ID;
};
const char AAAMDAttributes::ID = 0;

struct AAAMDWorkGroupSize
    : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAAMDWorkGroupSize(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAAMDWorkGroupSize &createForPosition(const IRPosition &IRP,
                                               Attributor &A);

  /// See AbstractAttribute::getName().
  const std::string getName() const override { return "AAAMDWorkGroupSize"; }

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
const char AAAMDWorkGroupSize::ID = 0;

struct AAAMDWorkGroupSizeFunction : public AAAMDWorkGroupSize {
  AAAMDWorkGroupSizeFunction(const IRPosition &IRP, Attributor &A)
      : AAAMDWorkGroupSize(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    CallingConv::ID CC = F->getCallingConv();

    if (CC != CallingConv::AMDGPU_KERNEL)
      return;

    bool InitialValue = false;
    if (F->hasFnAttribute("uniform-work-group-size"))
      InitialValue = F->getFnAttribute("uniform-work-group-size")
                         .getValueAsString()
                         .equals("true");

    if (InitialValue)
      indicateOptimisticFixpoint();
    else
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << "[AAAMDWorkGroupSize] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << "\n");

      const auto &CallerInfo = A.getAAFor<AAAMDWorkGroupSize>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);

      Change = Change | clampStateAndIndicateChange(this->getState(),
                                                    CallerInfo.getState());

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      indicatePessimisticFixpoint();

    return Change;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    AttrList.push_back(Attribute::get(Ctx, "uniform-work-group-size",
                                      getAssumed() ? "true" : "false"));
    return IRAttributeManifest::manifestAttrs(A, getIRPosition(), AttrList,
                                              /* ForceReplace */ true);
  }

  bool isValidState() const override {
    // This state is always valid, even when the state is false.
    return true;
  }

  const std::string getAsStr() const override {
    return "AMDWorkGroupSize[" + std::to_string(getAssumed()) + "]";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

AAAMDWorkGroupSize &AAAMDWorkGroupSize::createForPosition(const IRPosition &IRP,
                                                          Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAAMDWorkGroupSizeFunction(IRP, A);
  llvm_unreachable("AAAMDWorkGroupSize is only valid for function position");
}

struct AAAMDAttributesFunction : public AAAMDAttributes {
  AAAMDAttributesFunction(const IRPosition &IRP, Attributor &A)
      : AAAMDAttributes(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    CallingConv::ID CC = F->getCallingConv();
    bool CallingConvSupportsAllImplicits = (CC != CallingConv::AMDGPU_Gfx);

    // Don't add attributes to instrinsics
    if (F->isIntrinsic()) {
      indicatePessimisticFixpoint();
      return;
    }

    // Ignore functions with graphics calling conventions, these are currently
    // not allowed to have kernel arguments.
    if (AMDGPU::isGraphics(F->getCallingConv())) {
      indicatePessimisticFixpoint();
      return;
    }

    for (StringRef Attr : ImplicitAttrNames) {
      if (F->hasFnAttribute(Attr))
        Attributes.insert(Attr);
    }

    // TODO: We shouldn't need this in the future.
    if (CallingConvSupportsAllImplicits &&
        F->hasAddressTaken(nullptr, true, true, true)) {
      for (StringRef AttrName : ImplicitAttrNames) {
        Attributes.insert(AttrName);
      }
    }
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAssociatedFunction();
    ChangeStatus Change = ChangeStatus::UNCHANGED;
    bool IsNonEntryFunc = !AMDGPU::isEntryFunctionCC(F->getCallingConv());
    CallingConv::ID CC = F->getCallingConv();
    bool CallingConvSupportsAllImplicits = (CC != CallingConv::AMDGPU_Gfx);
    auto &InfoCache = static_cast<AMDGPUInformationCache &>(A.getInfoCache());

    auto AddAttribute = [&](StringRef AttrName) {
      if (Attributes.insert(AttrName).second)
        Change = ChangeStatus::CHANGED;
    };

    // Check for Intrinsics and propagate attributes.
    const AACallEdges &AAEdges = A.getAAFor<AACallEdges>(
        *this, this->getIRPosition(), DepClassTy::REQUIRED);

    // We have to assume that we can reach a function with these attributes.
    // We do not consider inline assembly as a unknown callee.
    if (CallingConvSupportsAllImplicits && AAEdges.hasNonAsmUnknownCallee()) {
      for (StringRef AttrName : ImplicitAttrNames) {
        AddAttribute(AttrName);
      }
    }

    bool NeedsQueuePtr = false;
    bool HasCall = false;
    for (Function *Callee : AAEdges.getOptimisticEdges()) {
      Intrinsic::ID IID = Callee->getIntrinsicID();
      if (IID != Intrinsic::not_intrinsic) {
        if (!IsNonEntryFunc && IID == Intrinsic::amdgcn_kernarg_segment_ptr) {
          AddAttribute("amdgpu-kernarg-segment-ptr");
          continue;
        }

        bool NonKernelOnly = false;
        StringRef AttrName =
            intrinsicToAttrName(IID, NonKernelOnly, NeedsQueuePtr);

        if (!AttrName.empty() && (IsNonEntryFunc || !NonKernelOnly))
          AddAttribute(AttrName);

        continue;
      }

      HasCall = true;
      const AAAMDAttributes &AAAMD = A.getAAFor<AAAMDAttributes>(
          *this, IRPosition::function(*Callee), DepClassTy::REQUIRED);
      const DenseSet<StringRef> &CalleeAttributes = AAAMD.getAttributes();
      // Propagate implicit attributes from called function.
      for (StringRef AttrName : ImplicitAttrNames)
        if (CalleeAttributes.count(AttrName))
          AddAttribute(AttrName);
    }

    HasCall |= AAEdges.hasUnknownCallee();
    if (!IsNonEntryFunc && HasCall)
      AddAttribute("amdgpu-calls");

    // Check the function body.
    auto CheckAlloca = [&](Instruction &I) {
      AddAttribute("amdgpu-stack-objects");
      return false;
    };

    bool UsedAssumedInformation = false;
    A.checkForAllInstructions(CheckAlloca, *this, {Instruction::Alloca},
                              UsedAssumedInformation);

    // If we found that we need amdgpu-queue-ptr, nothing else to do.
    if (NeedsQueuePtr || Attributes.count("amdgpu-queue-ptr")) {
      AddAttribute("amdgpu-queue-ptr");
      return Change;
    }

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

    // amdgpu-queue-ptr is not needed if aperture regs is present.
    if (!HasApertureRegs)
      A.checkForAllInstructions(CheckAddrSpaceCasts, *this,
                                {Instruction::AddrSpaceCast},
                                UsedAssumedInformation);

    // If we found  that we need amdgpu-queue-ptr, nothing else to do.
    if (NeedsQueuePtr) {
      AddAttribute("amdgpu-queue-ptr");
      return Change;
    }

    if (!IsNonEntryFunc && HasApertureRegs)
      return Change;

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        for (const Use &U : I.operands()) {
          if (const auto *C = dyn_cast<Constant>(U)) {
            if (InfoCache.needsQueuePtr(C, *F)) {
              AddAttribute("amdgpu-queue-ptr");
              return Change;
            }
          }
        }
      }
    }

    return Change;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    for (StringRef AttrName : Attributes)
      AttrList.push_back(Attribute::get(Ctx, AttrName));

    return IRAttributeManifest::manifestAttrs(A, getIRPosition(), AttrList,
                                              /* ForceReplace */ true);
  }

  const std::string getAsStr() const override {
    return "AMDInfo[" + std::to_string(Attributes.size()) + "]";
  }

  const DenseSet<StringRef> &getAttributes() const override {
    return Attributes;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

private:
  DenseSet<StringRef> Attributes;
};

AAAMDAttributes &AAAMDAttributes::createForPosition(const IRPosition &IRP,
                                                    Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAAMDAttributesFunction(IRP, A);
  llvm_unreachable("AAAMDAttributes is only valid for function position");
}

class AMDGPUAttributor : public ModulePass {
public:
  AMDGPUAttributor() : ModulePass(ID) {}

  /// doInitialization - Virtual method overridden by subclasses to do
  /// any necessary initialization before any pass is run.
  bool doInitialization(Module &) override {
    auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
    if (!TPC)
      report_fatal_error("TargetMachine is required");

    TM = &TPC->getTM<TargetMachine>();
    return false;
  }

  bool runOnModule(Module &M) override {
    SetVector<Function *> Functions;
    AnalysisGetter AG;
    for (Function &F : M)
      Functions.insert(&F);

    CallGraphUpdater CGUpdater;
    BumpPtrAllocator Allocator;
    AMDGPUInformationCache InfoCache(M, AG, Allocator, nullptr, *TM);
    Attributor A(Functions, InfoCache, CGUpdater);

    for (Function &F : M) {
      A.getOrCreateAAFor<AAAMDAttributes>(IRPosition::function(F));
      A.getOrCreateAAFor<AAAMDWorkGroupSize>(IRPosition::function(F));
    }

    ChangeStatus Change = A.run();
    return Change == ChangeStatus::CHANGED;
  }

  StringRef getPassName() const override { return "AMDGPU Attributor"; }
  TargetMachine *TM;
  static char ID;
};

char AMDGPUAttributor::ID = 0;

Pass *llvm::createAMDGPUAttributorPass() { return new AMDGPUAttributor(); }
INITIALIZE_PASS(AMDGPUAttributor, DEBUG_TYPE, "AMDGPU Attributor", false, false)
