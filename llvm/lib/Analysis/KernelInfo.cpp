//===- KernelInfo.cpp - Kernel Analysis -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the KernelInfoPrinter class used to emit remarks about
// function properties from a GPU kernel.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/KernelInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

#define DEBUG_TYPE "kernel-info"

namespace {

/// Data structure holding function info for kernels.
class KernelInfo {
  void updateForBB(const BasicBlock &BB, BlockFrequencyInfo &BFI,
                   OptimizationRemarkEmitter &ORE);

public:
  static void emitKernelInfo(Function &F, FunctionAnalysisManager &FAM,
                             TargetMachine *TM);

  /// Whether the function has external linkage and is not a kernel function.
  bool ExternalNotKernel = false;

  /// Launch bounds.
  SmallVector<std::pair<StringRef, int64_t>> LaunchBounds;

  /// The number of alloca instructions inside the function, the number of those
  /// with allocation sizes that cannot be determined at compile time, and the
  /// sum of the sizes that can be.
  ///
  /// With the current implementation for at least some GPU archs,
  /// AllocasDyn > 0 might not be possible, but we report AllocasDyn anyway in
  /// case the implementation changes.
  int64_t Allocas = 0;
  int64_t AllocasDyn = 0;
  int64_t AllocasStaticSizeSum = 0;

  /// Number of direct/indirect calls (anything derived from CallBase).
  int64_t DirectCalls = 0;
  int64_t IndirectCalls = 0;

  /// Number of direct calls made from this function to other functions
  /// defined in this module.
  int64_t DirectCallsToDefinedFunctions = 0;

  /// Number of direct calls to inline assembly.
  int64_t InlineAssemblyCalls = 0;

  /// Number of calls of type InvokeInst.
  int64_t Invokes = 0;

  /// Target-specific flat address space.
  unsigned FlatAddrspace;

  /// Number of flat address space memory accesses (via load, store, etc.).
  int64_t FlatAddrspaceAccesses = 0;

  /// Estimate of the number of floating point operations typically executed
  /// based on any available profile data.  If no profile data is available, the
  /// count is zero.
  uint64_t ProfileFloatingPointOpCount = 0;

  /// Estimate of the number bytes of floating point memory typically moved
  /// (e.g., load or store) based on any available profile data.  If no profile
  /// data is available, the count is zero.  LLVM memory access operations
  /// (e.g., llvm.memcpy.*, cmpxchg) that are always encoded as operating on
  /// integer types and never on floating point types are not included.
  uint64_t ProfileFloatingPointBytesMoved = 0;
};

} // end anonymous namespace

// For the purposes of KernelInfo::ProfileFloatingPointOpCount, should the
// specified Instruction be considered a floating point operation?  If so,
// return the floating point type and a multiplier for its FLOP count.
// Otherwise, return std::nullopt.
//
// TODO: Does this correctly identify floating point operations we care about?
// For example, we skip phi even when it returns a floating point value, and
// load is covered by KernelInfo::ProfileFloatingPointBytesMoved instead.  Is
// there anything missing that should be covered here?  Is there anything else
// that we should exclude?  For example, at least for AMD GPU, there are
// floating point instruction patterns (e.g., fmul with one operand in some
// category of immediate) that lower to instructions that do not trigger AMD's
// floating point hardware counters.  Should we somehow query target-specific
// lowering to exclude such cases?
static std::optional<std::pair<Type *, unsigned>>
getFloatingPointOp(const Instruction &I) {
  if (const AtomicRMWInst *At = dyn_cast<AtomicRMWInst>(&I)) {
    if (At->isFloatingPointOperation())
      return std::make_pair(At->getType(), 1);
    return std::nullopt;
  }
  if (const CastInst *CI = dyn_cast<CastInst>(&I)) {
    Type *SrcTy = CI->getSrcTy();
    Type *DestTy = CI->getDestTy();
    // For AMD GPU, conversions between fp and integer types where either is not
    // 64-bit lower to instructions that do not trigger AMD's floating point
    // hardware counters.  TODO: Is that true for all archs, all non-64-bit
    // floating point types, and all non-64-bit integer types?  On AMD GPU, we
    // have checked 64 vs. 32 and 32 vs. 32 so far.
    if (SrcTy->getScalarSizeInBits() != 64 ||
        DestTy->getScalarSizeInBits() != 64)
      return std::nullopt;
    // For AMD GPU, uitofp and sitofp lower to FADD instructions.  TODO: Is that
    // true for all archs?
    if (isa<UIToFPInst>(I) || isa<SIToFPInst>(I))
      return std::make_pair(DestTy, 1);
    // For AMD GPU, fptoui and fptosi lower to FMA instructions.  Thus, as for
    // FMA instructions below, we mutliply by 2.  TODO: Is that true for all
    // archs?
    if (isa<FPToUIInst>(I) || isa<FPToSIInst>(I))
      return std::make_pair(SrcTy, 2);
    return std::nullopt;
  }
  Type *Ty = I.getType();
  if (!Ty->isFPOrFPVectorTy())
    return std::nullopt;
  if (I.isBinaryOp() || I.isUnaryOp()) {
    switch (I.getOpcode()) {
    // For AMD GPU, fneg lowers to instructions that do not trigger AMD's
    // floating point hardware counters.  TODO: Is that true for all archs and
    // all floating point types?  On AMD GPU, we have check 64 bit.
    case Instruction::FNeg:
      return std::nullopt;
    // This multiplier is based on AMD hardware fp counters for fdiv:
    // - SQ_INSTS_VALU_FMA_F64   = 6*2
    // - SQ_INSTS_VALU_MUL_F64   = 1
    // - SQ_INSTS_VALU_TRANS_F64 = 1
    // TODO: Is that true for all archs and all floating point types?  On AMD
    // GPU, we have checked 64 bit.  Moreover, this is surely brittle.  What if
    // the implementation changes?
    case Instruction::FDiv:
      return std::make_pair(Ty, 14);
    }
    return std::make_pair(Ty, 1);
  }
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    // For AMD GPU, these lower to instructions that do not trigger AMD's
    // floating point hardware counters.  TODO: Is that true for all archs and
    // all floating point types?  On AMD GPU, we have checked 64 bit.
    case Intrinsic::copysign:
    case Intrinsic::fabs:
    case Intrinsic::floor:
    case Intrinsic::ldexp:
    case Intrinsic::minnum:
    case Intrinsic::rint:
      return std::nullopt;
    // For FMA instructions, we mimic AMD's rocprofiler-compute, which
    // multiplies SQ_INSTS_VALU_FMA_* counts by 2.
    case Intrinsic::fmuladd:
    case Intrinsic::fma:
      return std::make_pair(Ty, 2);
    // This multiplier is based on AMD hardware fp counters for this intrinsic:
    // - SQ_INSTS_VALU_FMA_F64   = 7*2
    // - SQ_INSTS_VALU_MUL_F64   = 2
    // - SQ_INSTS_VALU_TRANS_F64 = 1
    // TODO: Is that true for all archs and all floating point types?  On AMD
    // GPU, we have check 64 bit.  Moreover, this is surely brittle.  What if
    // the implementation changes?
    case Intrinsic::sqrt:
      return std::make_pair(Ty, 17);
    default:
      return std::make_pair(Ty, 1);
    }
  }
  return std::nullopt;
}

static void identifyCallee(OptimizationRemark &R, const Module *M,
                           const Value *V, StringRef Kind = "") {
  SmallString<100> Name; // might be function name or asm expression
  if (const Function *F = dyn_cast<Function>(V)) {
    if (auto *SubProgram = F->getSubprogram()) {
      if (SubProgram->isArtificial())
        R << "artificial ";
      Name = SubProgram->getName();
    }
  }
  if (Name.empty()) {
    raw_svector_ostream OS(Name);
    V->printAsOperand(OS, /*PrintType=*/false, M);
  }
  if (!Kind.empty())
    R << Kind << " ";
  R << "'" << Name << "'";
}

static void identifyFunction(OptimizationRemark &R, const Function &F) {
  identifyCallee(R, F.getParent(), &F, "function");
}

static void identifyInstruction(OptimizationRemark &R, const Instruction &I) {
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
    R << "'" << II->getCalledFunction()->getName() << "' call";
  else
    R << "'" << I.getOpcodeName() << "'";
  if (!I.getType()->isVoidTy()) {
    SmallString<20> Name;
    raw_svector_ostream OS(Name);
    I.printAsOperand(OS, /*PrintType=*/false, I.getModule());
    R << " ('" << Name << "')";
  }
}

static void remarkAlloca(OptimizationRemarkEmitter &ORE, const Function &Caller,
                         const AllocaInst &Alloca,
                         TypeSize::ScalarTy StaticSize) {
  ORE.emit([&] {
    StringRef DbgName;
    DebugLoc Loc;
    bool Artificial = false;
    auto DVRs = findDVRDeclares(&const_cast<AllocaInst &>(Alloca));
    if (!DVRs.empty()) {
      const DbgVariableRecord &DVR = **DVRs.begin();
      DbgName = DVR.getVariable()->getName();
      Loc = DVR.getDebugLoc();
      Artificial = DVR.Variable->isArtificial();
    }
    OptimizationRemark R(DEBUG_TYPE, "Alloca", DiagnosticLocation(Loc),
                         Alloca.getParent());
    R << "in ";
    identifyFunction(R, Caller);
    R << ", ";
    if (Artificial)
      R << "artificial ";
    SmallString<20> ValName;
    raw_svector_ostream OS(ValName);
    Alloca.printAsOperand(OS, /*PrintType=*/false, Caller.getParent());
    R << "alloca ('" << ValName << "') ";
    if (!DbgName.empty())
      R << "for '" << DbgName << "' ";
    else
      R << "without debug info ";
    R << "with ";
    if (StaticSize)
      R << "static size of " << itostr(StaticSize) << " bytes";
    else
      R << "dynamic size";
    return R;
  });
}

static void remarkCall(OptimizationRemarkEmitter &ORE, const Function &Caller,
                       const CallBase &Call, StringRef CallKind,
                       StringRef RemarkKind) {
  ORE.emit([&] {
    OptimizationRemark R(DEBUG_TYPE, RemarkKind, &Call);
    R << "in ";
    identifyFunction(R, Caller);
    R << ", " << CallKind << ", callee is ";
    identifyCallee(R, Caller.getParent(), Call.getCalledOperand());
    return R;
  });
}

static void remarkFlatAddrspaceAccess(OptimizationRemarkEmitter &ORE,
                                      const Function &Caller,
                                      const Instruction &I) {
  ORE.emit([&] {
    OptimizationRemark R(DEBUG_TYPE, "FlatAddrspaceAccess", &I);
    R << "in ";
    identifyFunction(R, Caller);
    R << ", ";
    identifyInstruction(R, I);
    R << " accesses memory in flat address space";
    return R;
  });
}

static void
remarkFloatingPointOp(OptimizationRemarkEmitter &ORE, const Function &Caller,
                      const Instruction &I, Type *Ty, unsigned Multiplier,
                      std::optional<uint64_t> BlockProfileCount,
                      std::optional<uint64_t> BytesMoved = std::nullopt) {
  ORE.emit([&] {
    OptimizationRemark R(DEBUG_TYPE,
                         BytesMoved ? "ProfileFloatingPointBytesMoved"
                                    : "ProfileFloatingPointOpCount",
                         &I);
    R << "in ";
    identifyFunction(R, Caller);
    R << ", ";
    SmallString<10> TyName;
    raw_svector_ostream OS(TyName);
    Ty->print(OS);
    R << TyName << " ";
    identifyInstruction(R, I);
    if (BlockProfileCount) {
      if (BytesMoved)
        R << " moved " << itostr(*BytesMoved * *BlockProfileCount)
          << " fp bytes";
      else
        R << " executed " << utostr(*BlockProfileCount) << " flops";
      if (Multiplier != 1)
        R << " x " << utostr(Multiplier);
    } else {
      R << " has no profile data";
    }
    return R;
  });
}

void KernelInfo::updateForBB(const BasicBlock &BB, BlockFrequencyInfo &BFI,
                             OptimizationRemarkEmitter &ORE) {
  const Function &F = *BB.getParent();
  const Module &M = *F.getParent();
  const DataLayout &DL = M.getDataLayout();
  // TODO: Is AllowSynthetic what we want?
  std::optional<uint64_t> BlockProfileCount =
      BFI.getBlockProfileCount(&BB, /*AllowSynthetic=*/true);
  for (const Instruction &I : BB.instructionsWithoutDebug()) {
    auto HandleFloatingPointBytesMoved = [&]() {
      Type *Ty = I.getAccessType();
      if (!Ty || !Ty->isFPOrFPVectorTy())
        return;
      TypeSize::ScalarTy Size = DL.getTypeStoreSize(Ty).getFixedValue();
      ProfileFloatingPointBytesMoved += BlockProfileCount.value_or(0) * Size;
      remarkFloatingPointOp(ORE, F, I, Ty, /*Multiplier=*/1, BlockProfileCount,
                            Size);
    };
    if (const AllocaInst *Alloca = dyn_cast<AllocaInst>(&I)) {
      ++Allocas;
      TypeSize::ScalarTy StaticSize = 0;
      if (std::optional<TypeSize> Size = Alloca->getAllocationSize(DL)) {
        StaticSize = Size->getFixedValue();
        assert(StaticSize <=
               (TypeSize::ScalarTy)std::numeric_limits<int64_t>::max());
        AllocasStaticSizeSum += StaticSize;
      } else {
        ++AllocasDyn;
      }
      remarkAlloca(ORE, F, *Alloca, StaticSize);
    } else if (const CallBase *Call = dyn_cast<CallBase>(&I)) {
      SmallString<40> CallKind;
      SmallString<40> RemarkKind;
      if (Call->isIndirectCall()) {
        ++IndirectCalls;
        CallKind += "indirect";
        RemarkKind += "Indirect";
      } else {
        ++DirectCalls;
        CallKind += "direct";
        RemarkKind += "Direct";
      }
      if (isa<InvokeInst>(Call)) {
        ++Invokes;
        CallKind += " invoke";
        RemarkKind += "Invoke";
      } else {
        CallKind += " call";
        RemarkKind += "Call";
      }
      if (!Call->isIndirectCall()) {
        if (const Function *Callee = Call->getCalledFunction()) {
          if (!Callee->isIntrinsic() && !Callee->isDeclaration()) {
            ++DirectCallsToDefinedFunctions;
            CallKind += " to defined function";
            RemarkKind += "ToDefinedFunction";
          }
        } else if (Call->isInlineAsm()) {
          ++InlineAssemblyCalls;
          CallKind += " to inline assembly";
          RemarkKind += "ToInlineAssembly";
        }
      }
      remarkCall(ORE, F, *Call, CallKind, RemarkKind);
      if (const AnyMemIntrinsic *MI = dyn_cast<AnyMemIntrinsic>(Call)) {
        if (MI->getDestAddressSpace() == FlatAddrspace) {
          ++FlatAddrspaceAccesses;
          remarkFlatAddrspaceAccess(ORE, F, I);
        } else if (const AnyMemTransferInst *MT =
                       dyn_cast<AnyMemTransferInst>(MI)) {
          if (MT->getSourceAddressSpace() == FlatAddrspace) {
            ++FlatAddrspaceAccesses;
            remarkFlatAddrspaceAccess(ORE, F, I);
          }
        }
        // llvm.memcpy.*, llvm.memset.*, etc. are encoded as operating on
        // integer types not floating point types, so
        // HandleFloatingPointBytesMoved is useless here.
      }
    } else if (const LoadInst *Load = dyn_cast<LoadInst>(&I)) {
      if (Load->getPointerAddressSpace() == FlatAddrspace) {
        ++FlatAddrspaceAccesses;
        remarkFlatAddrspaceAccess(ORE, F, I);
      }
      HandleFloatingPointBytesMoved();
    } else if (const StoreInst *Store = dyn_cast<StoreInst>(&I)) {
      if (Store->getPointerAddressSpace() == FlatAddrspace) {
        ++FlatAddrspaceAccesses;
        remarkFlatAddrspaceAccess(ORE, F, I);
      }
      HandleFloatingPointBytesMoved();
    } else if (const AtomicRMWInst *At = dyn_cast<AtomicRMWInst>(&I)) {
      if (At->getPointerAddressSpace() == FlatAddrspace) {
        ++FlatAddrspaceAccesses;
        remarkFlatAddrspaceAccess(ORE, F, I);
      }
      HandleFloatingPointBytesMoved();
    } else if (const AtomicCmpXchgInst *At = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (At->getPointerAddressSpace() == FlatAddrspace) {
        ++FlatAddrspaceAccesses;
        remarkFlatAddrspaceAccess(ORE, F, I);
      }
      // cmpxchg is encoded as operating on integer types not floating point
      // types, so HandleFloatingPointBytesMoved is useless here.
    }
    if (auto Op = getFloatingPointOp(I)) {
      Type *Ty;
      unsigned Multiplier;
      std::tie(Ty, Multiplier) = *Op;
      ProfileFloatingPointOpCount += Multiplier * BlockProfileCount.value_or(0);
      remarkFloatingPointOp(ORE, F, I, Ty, Multiplier, BlockProfileCount);
    }
  }
}

static std::string toString(bool Val) { return itostr(Val); }
static std::string toString(int64_t Val) { return itostr(Val); }
static std::string toString(uint64_t Val) { return utostr(Val); }

template <typename T>
void remarkProperty(OptimizationRemarkEmitter &ORE, const Function &F,
                    StringRef Name, T Val) {
  ORE.emit([&] {
    OptimizationRemark R(DEBUG_TYPE, Name, &F);
    R << "in ";
    identifyFunction(R, F);
    R << ", " << Name << " = " << toString(Val);
    return R;
  });
}

static std::optional<int64_t> parseFnAttrAsInteger(Function &F,
                                                   StringRef Name) {
  if (!F.hasFnAttribute(Name))
    return std::nullopt;
  return F.getFnAttributeAsParsedInteger(Name);
}

void KernelInfo::emitKernelInfo(Function &F, FunctionAnalysisManager &FAM,
                                TargetMachine *TM) {
  KernelInfo KI;
  TargetTransformInfo &TheTTI = FAM.getResult<TargetIRAnalysis>(F);
  BlockFrequencyInfo &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
  KI.FlatAddrspace = TheTTI.getFlatAddressSpace();

  // Record function properties.
  KI.ExternalNotKernel = F.hasExternalLinkage() && !F.hasKernelCallingConv();
  for (StringRef Name : {"omp_target_num_teams", "omp_target_thread_limit"}) {
    if (auto Val = parseFnAttrAsInteger(F, Name))
      KI.LaunchBounds.push_back({Name, *Val});
  }
  TheTTI.collectKernelLaunchBounds(F, KI.LaunchBounds);

  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  for (const auto &BB : F)
    KI.updateForBB(BB, BFI, ORE);

#define REMARK_PROPERTY(PROP_NAME)                                             \
  remarkProperty(ORE, F, #PROP_NAME, KI.PROP_NAME)
  REMARK_PROPERTY(ExternalNotKernel);
  for (auto LB : KI.LaunchBounds)
    remarkProperty(ORE, F, LB.first, LB.second);
  REMARK_PROPERTY(Allocas);
  REMARK_PROPERTY(AllocasStaticSizeSum);
  REMARK_PROPERTY(AllocasDyn);
  REMARK_PROPERTY(DirectCalls);
  REMARK_PROPERTY(IndirectCalls);
  REMARK_PROPERTY(DirectCallsToDefinedFunctions);
  REMARK_PROPERTY(InlineAssemblyCalls);
  REMARK_PROPERTY(Invokes);
  REMARK_PROPERTY(FlatAddrspaceAccesses);
  REMARK_PROPERTY(ProfileFloatingPointOpCount);
  REMARK_PROPERTY(ProfileFloatingPointBytesMoved);
#undef REMARK_PROPERTY
}

PreservedAnalyses KernelInfoPrinter::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  // Skip it if remarks are not enabled as it will do nothing useful.
  if (F.getContext().getDiagHandlerPtr()->isPassedOptRemarkEnabled(DEBUG_TYPE))
    KernelInfo::emitKernelInfo(F, AM, TM);
  return PreservedAnalyses::all();
}
