//===-- NVPTXLowerArgs.cpp - Lower arguments ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Arguments to kernel functions are passed via param space, which imposes
// certain restrictions:
// http://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces
//
// Kernel parameters are read-only and accessible only via ld.param
// instruction, directly or via a pointer.
//
// Copying a byval struct into local memory in IR allows us to enforce
// the param space restrictions, gives the rest of IR a pointer w/o
// param space restrictions, and gives us an opportunity to eliminate
// the copy.
//
// This pass lowers byval parameters of kernel functions. It rewrites the
// kernel's signature so that each byval argument is declared directly as a
// pointer in the param address space (`ptr addrspace(101)`), then adjusts the
// body to match. The parameter symbols occupy this space when lowered during
// ISel, so making the IR type honest avoids the need for a cast or intrinsic to
// reinterpret a generic pointer as a param-space pointer.
//
// This pass uses 1 of 3 possible strategies to lower byval parameters:
//
// 1. Direct readonly nocapture uses: If we can trace through all the uses and
//    we can convert them all to param AS, then we'll do this. This is useful
//    for pre-SM70 targets where cvta.param is not available.
//
// 2. Grid constant: If the argument is a grid constant (and the target supports
//    cvta.param), we can cast back to generic address space to use the pointer
//    directly.
//
// 3. Local copy: If we can't trace through all the uses and we can't convert
//    them all to param AS, then we'll create a local copy of the argument in
//    local memory. This is useful for arguments that are mutated.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXUtilities.h"
#include "NVVMProperties.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"

#define DEBUG_TYPE "nvptx-lower-args"

using namespace llvm;
using namespace NVPTXAS;

namespace {
class NVPTXLowerArgsLegacyPass : public ModulePass {
  bool runOnModule(Module &M) override;

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerArgsLegacyPass() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "Lower pointer arguments of CUDA kernels";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};
} // namespace

char NVPTXLowerArgsLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(NVPTXLowerArgsLegacyPass, "nvptx-lower-args",
                      "Lower arguments (NVPTX)", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(NVPTXLowerArgsLegacyPass, "nvptx-lower-args",
                    "Lower arguments (NVPTX)", false, false)

/// Recursively convert the users of a param to the param address space.
static void convertToParamAS(ArrayRef<Use *> OldUses, Value *Param) {
  struct IP {
    Use *OldUse;
    Value *NewParam;
  };

  const auto CloneInstInParamAS = [](const IP &I) -> Value * {
    auto *OldInst = cast<Instruction>(I.OldUse->getUser());
    if (auto *LI = dyn_cast<LoadInst>(OldInst)) {
      LI->setOperand(0, I.NewParam);
      return LI;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(OldInst)) {
      SmallVector<Value *, 4> Indices(GEP->indices());
      auto *NewGEP = GetElementPtrInst::Create(
          GEP->getSourceElementType(), I.NewParam, Indices, GEP->getName(),
          GEP->getIterator());
      NewGEP->setNoWrapFlags(GEP->getNoWrapFlags());
      return NewGEP;
    }
    if (auto *BC = dyn_cast<BitCastInst>(OldInst)) {
      auto *NewBCType =
          PointerType::get(BC->getContext(), ADDRESS_SPACE_ENTRY_PARAM);
      return BitCastInst::Create(BC->getOpcode(), I.NewParam, NewBCType,
                                 BC->getName(), BC->getIterator());
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(OldInst)) {
      assert(ASC->getDestAddressSpace() == ADDRESS_SPACE_ENTRY_PARAM);
      (void)ASC;
      // Just pass through the argument, the old ASC is no longer needed.
      return I.NewParam;
    }
    if (auto *MI = dyn_cast<MemTransferInst>(OldInst)) {
      if (MI->getRawSource() == I.OldUse->get()) {
        // convert to memcpy/memmove from param space.
        IRBuilder<> Builder(OldInst);
        Intrinsic::ID ID = MI->getIntrinsicID();

        CallInst *B = Builder.CreateMemTransferInst(
            ID, MI->getRawDest(), MI->getDestAlign(), I.NewParam,
            MI->getSourceAlign(), MI->getLength(), MI->isVolatile());
        for (unsigned I : {0, 1})
          if (uint64_t Bytes = MI->getParamDereferenceableBytes(I))
            B->addDereferenceableParamAttr(I, Bytes);
        return B;
      }
    }

    llvm_unreachable("Unsupported instruction");
  };

  auto ItemsToConvert =
      map_to_vector(OldUses, [=](Use *U) -> IP { return {U, Param}; });
  SmallVector<Instruction *> InstructionsToDelete;

  while (!ItemsToConvert.empty()) {
    IP I = ItemsToConvert.pop_back_val();
    Value *NewInst = CloneInstInParamAS(I);
    Instruction *OldInst = cast<Instruction>(I.OldUse->getUser());

    if (NewInst && NewInst != OldInst) {
      // We've created a new instruction. Queue users of the old instruction to
      // be converted and the instruction itself to be deleted. We can't delete
      // the old instruction yet, because it's still in use by a load somewhere.
      for (Use &U : OldInst->uses())
        ItemsToConvert.push_back({&U, NewInst});

      InstructionsToDelete.push_back(OldInst);
    }
  }

  // Now we know that all argument loads are using addresses in parameter space
  // and we can finally remove the old instructions in generic AS. Instructions
  // scheduled for removal should be processed in reverse order so the ones
  // closest to the load are deleted first. Otherwise they may still be in use.
  // E.g if we have Value = Load(BitCast(GEP(arg))), InstructionsToDelete will
  // have {GEP,BitCast}. GEP can't be deleted first, because it's still used by
  // the BitCast.
  for (Instruction *I : llvm::reverse(InstructionsToDelete))
    I->eraseFromParent();
}

namespace {
struct ArgUseChecker : PtrUseVisitor<ArgUseChecker> {
  using Base = PtrUseVisitor<ArgUseChecker>;
  // Set of phi/select instructions using the Arg
  SmallPtrSet<Instruction *, 4> Conditionals;

  ArgUseChecker(const DataLayout &DL) : PtrUseVisitor(DL) {}

  PtrInfo visitArgPtr(Argument &A) {
    assert(A.getType()->isPointerTy());
    IntegerType *IntIdxTy = cast<IntegerType>(DL.getIndexType(A.getType()));
    IsOffsetKnown = false;
    Offset = APInt(IntIdxTy->getBitWidth(), 0);
    PI.reset();

    LLVM_DEBUG(dbgs() << "Checking Argument " << A << "\n");
    // Enqueue the uses of this pointer.
    enqueueUsers(A);

    // Visit all the uses off the worklist until it is empty.
    // Note that unlike PtrUseVisitor we intentionally do not track offsets.
    // We're only interested in how we use the pointer.
    while (!(Worklist.empty() || PI.isAborted())) {
      UseToVisit ToVisit = Worklist.pop_back_val();
      U = ToVisit.UseAndIsOffsetKnown.getPointer();
      Instruction *I = cast<Instruction>(U->getUser());
      LLVM_DEBUG(dbgs() << "Processing " << *I << "\n");
      Base::visit(I);
    }
    if (PI.isEscaped())
      LLVM_DEBUG(dbgs() << "Argument pointer escaped: " << *PI.getEscapingInst()
                        << "\n");
    else if (PI.isAborted())
      LLVM_DEBUG(dbgs() << "Pointer use needs a copy: " << *PI.getAbortingInst()
                        << "\n");
    LLVM_DEBUG(dbgs() << "Traversed " << Conditionals.size()
                      << " conditionals\n");
    return PI;
  }

  void visitStoreInst(StoreInst &SI) {
    // Storing the pointer escapes it.
    if (U->get() == SI.getValueOperand())
      return PI.setEscapedAndAborted(&SI);

    PI.setAborted(&SI);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // ASC to param space are no-ops and do not need a copy
    if (ASC.getDestAddressSpace() != ADDRESS_SPACE_ENTRY_PARAM)
      return PI.setEscapedAndAborted(&ASC);
    Base::visitAddrSpaceCastInst(ASC);
  }

  void visitPtrToIntInst(PtrToIntInst &I) { Base::visitPtrToIntInst(I); }

  void visitPHINodeOrSelectInst(Instruction &I) {
    assert(isa<PHINode>(I) || isa<SelectInst>(I));
    enqueueUsers(I);
    Conditionals.insert(&I);
  }
  // PHI and select just pass through the pointers.
  void visitPHINode(PHINode &PN) { visitPHINodeOrSelectInst(PN); }
  void visitSelectInst(SelectInst &SI) { visitPHINodeOrSelectInst(SI); }

  // memcpy/memmove are OK when the pointer is source. We can convert them to
  // AS-specific memcpy.
  void visitMemTransferInst(MemTransferInst &II) {
    if (*U == II.getRawDest())
      PI.setAborted(&II);
  }

  void visitMemSetInst(MemSetInst &II) { PI.setAborted(&II); }
}; // struct ArgUseChecker

// Create a local copy of the byval parameter \p Arg in an alloca, filled by a
// copy from \p ParamPtr (a pointer to the parameter), and replace all uses of
// \p Arg with the alloca. \p ParamPtr is either the natively param-space
// argument (when called from the signature rewrite) or the generic byval
// argument itself (when called early, before the signature has been rewritten).
void copyByValParam(Function &F, Argument &Arg, Value &ParamPtr) {
  LLVM_DEBUG(dbgs() << "Creating a local copy of " << Arg << "\n");
  Type *ByValType = Arg.getParamByValType();
  const DataLayout &DL = F.getDataLayout();
  IRBuilder<> IRB(&F.getEntryBlock().front());
  AllocaInst *AllocA = IRB.CreateAlloca(ByValType, nullptr, Arg.getName());
  // Set the alignment to alignment of the byval parameter. This is because,
  // later load/stores assume that alignment, and we are going to replace
  // the use of the byval parameter with this alloca instruction.
  AllocA->setAlignment(
      Arg.getParamAlign().value_or(DL.getPrefTypeAlign(ByValType)));
  Arg.replaceAllUsesWith(AllocA);

  // Be sure to propagate alignment to this copy; LLVM doesn't know that NVPTX
  // addrspacecast preserves alignment.  Since params are constant, this copy
  // is definitely not volatile.
  const auto ArgSize = *AllocA->getAllocationSize(DL);
  IRB.CreateMemCpy(AllocA, AllocA->getAlign(), &ParamPtr, AllocA->getAlign(),
                   ArgSize);
}
} // namespace

// Returns true if F has a byval argument not yet in the param address space.
// Such arguments are lowered exactly once, so one already in param space means
// the kernel has already been processed.
static bool kernelNeedsByValLowering(const Function &F) {
  return any_of(F.args(), [](const Argument &A) {
    return A.hasByValAttr() &&
           A.getType()->getPointerAddressSpace() != ADDRESS_SPACE_ENTRY_PARAM;
  });
}

// Lower the uses of a single kernel byval argument. \p OldArg is the original
// (generic) argument whose uses are being rewritten; \p NewParamArg is its
// replacement, natively in the param address space.
static void lowerKernelByValParam(Argument &OldArg, Argument &NewParamArg,
                                  Function &F, const bool HasCvtaParam) {
  assert(isKernelFunction(F));

  const DataLayout &DL = F.getDataLayout();
  IRBuilder<> IRB(&F.getEntryBlock().front());

  if (OldArg.use_empty())
    return;

  // (1) First check the easy case, if were able to trace through all the uses
  // and we can convert them all to param AS, then we'll do this.
  ArgUseChecker AUC(DL);
  ArgUseChecker::PtrInfo PI = AUC.visitArgPtr(OldArg);
  const bool ArgUseIsReadOnly = !(PI.isEscaped() || PI.isAborted());
  if (ArgUseIsReadOnly && AUC.Conditionals.empty()) {
    // Convert all loads and intermediate operations to use parameter AS and
    // skip creation of a local copy of the argument.
    SmallVector<Use *, 16> UsesToUpdate(make_pointer_range(OldArg.uses()));
    for (Use *U : UsesToUpdate)
      convertToParamAS(U, &NewParamArg);
    // This path does not replaceAllUsesWith the old argument, so any debug-info
    // uses would be left dangling and reset to poison when the old function is
    // erased. Point them at the new param-space argument instead.
    if (OldArg.isUsedByMetadata()) {
      SmallVector<DbgVariableRecord *, 4> DbgUsers;
      findDbgUsers(&OldArg, DbgUsers);
      for (DbgVariableRecord *DVR : DbgUsers)
        DVR->replaceVariableLocationOp(&OldArg, &NewParamArg);
    }
    return;
  }

  // (2) If the argument is grid constant, we get to use the pointer directly.
  if (HasCvtaParam && (ArgUseIsReadOnly || isParamGridConstant(OldArg))) {
    LLVM_DEBUG(dbgs() << "Using non-copy pointer to " << OldArg << "\n");

    // Cast the param-space argument to the generic address space. Because the
    // argument is natively in param space, this cast only ever goes
    // param -> generic and lowers to cvta.param; there is no inverse cast for
    // InferAddressSpaces to fold it away with.
    Value *GenericArg = IRB.CreateAddrSpaceCast(
        &NewParamArg, IRB.getPtrTy(ADDRESS_SPACE_GENERIC),
        OldArg.getName() + ".gen");

    OldArg.replaceAllUsesWith(GenericArg);
    return;
  }

  // (3) Otherwise we have to create a copy of the argument in local memory.
  copyByValParam(F, OldArg, NewParamArg);
}

// Rewrite a kernel's signature so that each byval argument is declared directly
// as a pointer in the param address space, then lower the body to match. This
// creates a new function, moves the body across, and erases \p F.
static void rewriteKernelByValSignature(Function &F, const bool HasCvtaParam) {
  LLVMContext &Ctx = F.getContext();
  FunctionType *FTy = F.getFunctionType();

  // Build the new signature: byval pointer arguments move to the param address
  // space; all other arguments are unchanged.
  SmallVector<Type *> Params(FTy->params());
  for (const Argument &Arg : F.args())
    if (Arg.hasByValAttr())
      Params[Arg.getArgNo()] = PointerType::get(Ctx, ADDRESS_SPACE_ENTRY_PARAM);

  Function *NF = Function::Create(
      FunctionType::get(FTy->getReturnType(), Params, FTy->isVarArg()),
      F.getLinkage(), F.getAddressSpace());
  NF->copyAttributesFrom(&F);
  NF->setComdat(F.getComdat());
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);

  // ISel reads the param symbol directly for kernel byval arguments; this is
  // valid because the signature rewrite above puts them in the param address
  // space. Mark them readonly: any mutation is redirected to a local copy
  // below, so the param itself is never written.
  for (Argument &NewArg : NF->args())
    if (NewArg.hasByValAttr())
      NewArg.addAttr(Attribute::ReadOnly);

  // Take over F's name and uses (e.g. @llvm.used, nvvm.annotations metadata),
  // then move the body across.
  F.replaceAllUsesWith(NF);
  NF->takeName(&F);
  NF->splice(NF->begin(), &F);

  // Remap arguments. Non-byval arguments keep their type and are replaced
  // directly; byval arguments change address space, so their uses are lowered
  // to operate on the new param-space argument.
  for (auto [OldArg, NewArg] : zip_equal(F.args(), NF->args())) {
    if (OldArg.hasByValAttr())
      lowerKernelByValParam(OldArg, NewArg, *NF, HasCvtaParam);
    else
      OldArg.replaceAllUsesWith(&NewArg);
    NewArg.takeName(&OldArg);
  }

  // Move function-level metadata (debug info, etc.) to the new function.
  NF->copyMetadata(&F, /*Offset=*/0);
  F.clearMetadata();

  F.eraseFromParent();
}

// =============================================================================
// Main function for this pass.
// =============================================================================
static bool processFunction(Function &F, NVPTXTargetMachine &TM) {
  if (!isKernelFunction(F) || F.isDeclaration())
    return false;

  // Skip kernels with no byval arguments, and those already lowered (byval
  // arguments sitting in the param address space).
  if (!kernelNeedsByValLowering(F))
    return false;

  LLVM_DEBUG(dbgs() << "Lowering kernel args of " << F.getName() << "\n");
  const NVPTXSubtarget *ST = TM.getSubtargetImpl(F);
  rewriteKernelByValSignature(F, ST->hasCvtaParam());
  return true;
}

static bool processModule(Module &M, NVPTXTargetMachine &TM) {
  bool Changed = false;
  for (Function &F : make_early_inc_range(M))
    Changed |= processFunction(F, TM);
  return Changed;
}

bool NVPTXLowerArgsLegacyPass::runOnModule(Module &M) {
  auto &TM = getAnalysis<TargetPassConfig>().getTM<NVPTXTargetMachine>();
  return processModule(M, TM);
}

ModulePass *llvm::createNVPTXLowerArgsPass() {
  return new NVPTXLowerArgsLegacyPass();
}

static bool copyFunctionByValArgs(Function &F) {
  LLVM_DEBUG(dbgs() << "Creating a copy of byval args of " << F.getName()
                    << "\n");
  bool Changed = false;
  if (isKernelFunction(F)) {
    for (Argument &Arg : F.args())
      if (Arg.hasByValAttr() && !isParamGridConstant(Arg)) {
        copyByValParam(F, Arg, Arg);
        Changed = true;
      }
  }
  return Changed;
}

PreservedAnalyses NVPTXCopyByValArgsPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  return copyFunctionByValArgs(F) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

PreservedAnalyses NVPTXLowerArgsPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto &NTM = static_cast<NVPTXTargetMachine &>(TM);
  bool Changed = processModule(M, NTM);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
