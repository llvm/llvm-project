//===-- NVPTXLowerArgs.cpp - Lower arguments ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// Arguments to kernel and device functions are passed via param space,
// which imposes certain restrictions:
// http://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces
//
// Kernel parameters are read-only and accessible only via ld.param
// instruction, directly or via a pointer.
//
// Device function parameters are directly accessible via
// ld.param/st.param, but taking the address of one returns a pointer
// to a copy created in local space which *can't* be used with
// ld.param/st.param.
//
// Copying a byval struct into local memory in IR allows us to enforce
// the param space restrictions, gives the rest of IR a pointer w/o
// param space restrictions, and gives us an opportunity to eliminate
// the copy.
//
// Pointer arguments to kernel functions need more work to be lowered:
//
// 1. Convert non-byval pointer arguments of CUDA kernels to pointers in the
//    global address space. This allows later optimizations to emit
//    ld.global.*/st.global.* for accessing these pointer arguments. For
//    example,
//
//    define void @foo(float* %input) {
//      %v = load float, float* %input, align 4
//      ...
//    }
//
//    becomes
//
//    define void @foo(float* %input) {
//      %input2 = addrspacecast float* %input to float addrspace(1)*
//      %input3 = addrspacecast float addrspace(1)* %input2 to float*
//      %v = load float, float* %input3, align 4
//      ...
//    }
//
//    Later, NVPTXInferAddressSpaces will optimize it to
//
//    define void @foo(float* %input) {
//      %input2 = addrspacecast float* %input to float addrspace(1)*
//      %v = load float, float addrspace(1)* %input2, align 4
//      ...
//    }
//
// 2. Convert byval kernel parameters to pointers in the param address space
//    (so that NVPTX emits ld/st.param).  Convert pointers *within* a byval
//    kernel parameter to pointers in the global address space. This allows
//    NVPTX to emit ld/st.global.
//
//    struct S {
//      int *x;
//      int *y;
//    };
//    __global__ void foo(S s) {
//      int *b = s.y;
//      // use b
//    }
//
//    "b" points to the global address space. In the IR level,
//
//    define void @foo(ptr byval %input) {
//      %b_ptr = getelementptr {ptr, ptr}, ptr %input, i64 0, i32 1
//      %b = load ptr, ptr %b_ptr
//      ; use %b
//    }
//
//    becomes
//
//    define void @foo({i32*, i32*}* byval %input) {
//      %b_param = addrspacecat ptr %input to ptr addrspace(101)
//      %b_ptr = getelementptr {ptr, ptr}, ptr addrspace(101) %b_param, i64 0, i32 1
//      %b = load ptr, ptr addrspace(101) %b_ptr
//      %b_global = addrspacecast ptr %b to ptr addrspace(1)
//      ; use %b_generic
//    }
//
//    Create a local copy of kernel byval parameters used in a way that *might* mutate
//    the parameter, by storing it in an alloca. Mutations to "grid_constant" parameters
//    are undefined behaviour, and don't require local copies.
//
//    define void @foo(ptr byval(%struct.s) align 4 %input) {
//       store i32 42, ptr %input
//       ret void
//    }
//
//    becomes
//
//    define void @foo(ptr byval(%struct.s) align 4 %input) #1 {
//      %input1 = alloca %struct.s, align 4
//      %input2 = addrspacecast ptr %input to ptr addrspace(101)
//      %input3 = load %struct.s, ptr addrspace(101) %input2, align 4
//      store %struct.s %input3, ptr %input1, align 4
//      store i32 42, ptr %input1, align 4
//      ret void
//    }
//
//    If %input were passed to a device function, or written to memory,
//    conservatively assume that %input gets mutated, and create a local copy.
//
//    Convert param pointers to grid_constant byval kernel parameters that are
//    passed into calls (device functions, intrinsics, inline asm), or otherwise
//    "escape" (into stores/ptrtoints) to the generic address space, using the
//    `nvvm.ptr.param.to.gen` intrinsic, so that NVPTX emits cvta.param
//    (available for sm70+)
//
//    define void @foo(ptr byval(%struct.s) %input) {
//      ; %input is a grid_constant
//      %call = call i32 @escape(ptr %input)
//      ret void
//    }
//
//    becomes
//
//    define void @foo(ptr byval(%struct.s) %input) {
//      %input1 = addrspacecast ptr %input to ptr addrspace(101)
//      ; the following intrinsic converts pointer to generic. We don't use an addrspacecast
//      ; to prevent generic -> param -> generic from getting cancelled out
//      %input1.gen = call ptr @llvm.nvvm.ptr.param.to.gen.p0.p101(ptr addrspace(101) %input1)
//      %call = call i32 @escape(ptr %input1.gen)
//      ret void
//    }
//
// TODO: merge this pass with NVPTXInferAddressSpaces so that other passes don't
// cancel the addrspacecast pair this pass emits.
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include <numeric>
#include <queue>

#define DEBUG_TYPE "nvptx-lower-args"

using namespace llvm;

namespace {
class NVPTXLowerArgsLegacyPass : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerArgsLegacyPass() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "Lower pointer arguments of CUDA kernels";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};
} // namespace

char NVPTXLowerArgsLegacyPass::ID = 1;

INITIALIZE_PASS_BEGIN(NVPTXLowerArgsLegacyPass, "nvptx-lower-args",
                      "Lower arguments (NVPTX)", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(NVPTXLowerArgsLegacyPass, "nvptx-lower-args",
                    "Lower arguments (NVPTX)", false, false)

// =============================================================================
// If the function had a byval struct ptr arg, say foo(%struct.x* byval %d),
// and we can't guarantee that the only accesses are loads,
// then add the following instructions to the first basic block:
//
// %temp = alloca %struct.x, align 8
// %tempd = addrspacecast %struct.x* %d to %struct.x addrspace(101)*
// %tv = load %struct.x addrspace(101)* %tempd
// store %struct.x %tv, %struct.x* %temp, align 8
//
// The above code allocates some space in the stack and copies the incoming
// struct from param space to local space.
// Then replace all occurrences of %d by %temp.
//
// In case we know that all users are GEPs or Loads, replace them with the same
// ones in parameter AS, so we can access them using ld.param.
// =============================================================================

// For Loads, replaces the \p OldUse of the pointer with a Use of the same
// pointer in parameter AS.
// For "escapes" (to memory, a function call, or a ptrtoint), cast the OldUse to
// generic using cvta.param.
static void convertToParamAS(Use *OldUse, Value *Param, bool HasCvtaParam,
                             bool IsGridConstant) {
  Instruction *I = dyn_cast<Instruction>(OldUse->getUser());
  assert(I && "OldUse must be in an instruction");
  struct IP {
    Use *OldUse;
    Instruction *OldInstruction;
    Value *NewParam;
  };
  SmallVector<IP> ItemsToConvert = {{OldUse, I, Param}};
  SmallVector<Instruction *> InstructionsToDelete;

  auto CloneInstInParamAS = [HasCvtaParam,
                             IsGridConstant](const IP &I) -> Value * {
    if (auto *LI = dyn_cast<LoadInst>(I.OldInstruction)) {
      LI->setOperand(0, I.NewParam);
      return LI;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I.OldInstruction)) {
      SmallVector<Value *, 4> Indices(GEP->indices());
      auto *NewGEP = GetElementPtrInst::Create(
          GEP->getSourceElementType(), I.NewParam, Indices, GEP->getName(),
          GEP->getIterator());
      NewGEP->setIsInBounds(GEP->isInBounds());
      return NewGEP;
    }
    if (auto *BC = dyn_cast<BitCastInst>(I.OldInstruction)) {
      auto *NewBCType = PointerType::get(BC->getContext(), ADDRESS_SPACE_PARAM);
      return BitCastInst::Create(BC->getOpcode(), I.NewParam, NewBCType,
                                 BC->getName(), BC->getIterator());
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(I.OldInstruction)) {
      assert(ASC->getDestAddressSpace() == ADDRESS_SPACE_PARAM);
      (void)ASC;
      // Just pass through the argument, the old ASC is no longer needed.
      return I.NewParam;
    }
    if (auto *MI = dyn_cast<MemTransferInst>(I.OldInstruction)) {
      if (MI->getRawSource() == I.OldUse->get()) {
        // convert to memcpy/memmove from param space.
        IRBuilder<> Builder(I.OldInstruction);
        Intrinsic::ID ID = MI->getIntrinsicID();

        CallInst *B = Builder.CreateMemTransferInst(
            ID, MI->getRawDest(), MI->getDestAlign(), I.NewParam,
            MI->getSourceAlign(), MI->getLength(), MI->isVolatile());
        for (unsigned I : {0, 1})
          if (uint64_t Bytes = MI->getParamDereferenceableBytes(I))
            B->addDereferenceableParamAttr(I, Bytes);
        return B;
      }
      // We may be able to handle other cases if the argument is
      // __grid_constant__
    }

    if (HasCvtaParam) {
      auto GetParamAddrCastToGeneric =
          [](Value *Addr, Instruction *OriginalUser) -> Value * {
        IRBuilder<> IRB(OriginalUser);
        Type *GenTy = IRB.getPtrTy(ADDRESS_SPACE_GENERIC);
        return IRB.CreateAddrSpaceCast(Addr, GenTy, Addr->getName() + ".gen");
      };
      auto *ParamInGenericAS =
          GetParamAddrCastToGeneric(I.NewParam, I.OldInstruction);

      // phi/select could use generic arg pointers w/o __grid_constant__
      if (auto *PHI = dyn_cast<PHINode>(I.OldInstruction)) {
        for (auto [Idx, V] : enumerate(PHI->incoming_values())) {
          if (V.get() == I.OldUse->get())
            PHI->setIncomingValue(Idx, ParamInGenericAS);
        }
      }
      if (auto *SI = dyn_cast<SelectInst>(I.OldInstruction)) {
        if (SI->getTrueValue() == I.OldUse->get())
          SI->setTrueValue(ParamInGenericAS);
        if (SI->getFalseValue() == I.OldUse->get())
          SI->setFalseValue(ParamInGenericAS);
      }

      // Escapes or writes can only use generic param pointers if
      // __grid_constant__ is in effect.
      if (IsGridConstant) {
        if (auto *CI = dyn_cast<CallInst>(I.OldInstruction)) {
          I.OldUse->set(ParamInGenericAS);
          return CI;
        }
        if (auto *SI = dyn_cast<StoreInst>(I.OldInstruction)) {
          // byval address is being stored, cast it to generic
          if (SI->getValueOperand() == I.OldUse->get())
            SI->setOperand(0, ParamInGenericAS);
          return SI;
        }
        if (auto *PI = dyn_cast<PtrToIntInst>(I.OldInstruction)) {
          if (PI->getPointerOperand() == I.OldUse->get())
            PI->setOperand(0, ParamInGenericAS);
          return PI;
        }
        // TODO: iIf we allow stores, we should allow memcpy/memset to
        // parameter, too.
      }
    }

    llvm_unreachable("Unsupported instruction");
  };

  while (!ItemsToConvert.empty()) {
    IP I = ItemsToConvert.pop_back_val();
    Value *NewInst = CloneInstInParamAS(I);

    if (NewInst && NewInst != I.OldInstruction) {
      // We've created a new instruction. Queue users of the old instruction to
      // be converted and the instruction itself to be deleted. We can't delete
      // the old instruction yet, because it's still in use by a load somewhere.
      for (Use &U : I.OldInstruction->uses())
        ItemsToConvert.push_back({&U, cast<Instruction>(U.getUser()), NewInst});

      InstructionsToDelete.push_back(I.OldInstruction);
    }
  }

  // Now we know that all argument loads are using addresses in parameter space
  // and we can finally remove the old instructions in generic AS.  Instructions
  // scheduled for removal should be processed in reverse order so the ones
  // closest to the load are deleted first. Otherwise they may still be in use.
  // E.g if we have Value = Load(BitCast(GEP(arg))), InstructionsToDelete will
  // have {GEP,BitCast}. GEP can't be deleted first, because it's still used by
  // the BitCast.
  for (Instruction *I : llvm::reverse(InstructionsToDelete))
    I->eraseFromParent();
}

// Adjust alignment of arguments passed byval in .param address space. We can
// increase alignment of such arguments in a way that ensures that we can
// effectively vectorize their loads. We should also traverse all loads from
// byval pointer and adjust their alignment, if those were using known offset.
// Such alignment changes must be conformed with parameter store and load in
// NVPTXTargetLowering::LowerCall.
static void adjustByValArgAlignment(Argument *Arg, Value *ArgInParamAS,
                                    const NVPTXTargetLowering *TLI) {
  Function *Func = Arg->getParent();
  Type *StructType = Arg->getParamByValType();
  const DataLayout &DL = Func->getDataLayout();

  const Align NewArgAlign =
      TLI->getFunctionParamOptimizedAlign(Func, StructType, DL);
  const Align CurArgAlign = Arg->getParamAlign().valueOrOne();

  if (CurArgAlign >= NewArgAlign)
    return;

  LLVM_DEBUG(dbgs() << "Try to use alignment " << NewArgAlign.value()
                    << " instead of " << CurArgAlign.value() << " for " << *Arg
                    << '\n');

  auto NewAlignAttr =
      Attribute::getWithAlignment(Func->getContext(), NewArgAlign);
  Arg->removeAttr(Attribute::Alignment);
  Arg->addAttr(NewAlignAttr);

  struct Load {
    LoadInst *Inst;
    uint64_t Offset;
  };

  struct LoadContext {
    Value *InitialVal;
    uint64_t Offset;
  };

  SmallVector<Load> Loads;
  std::queue<LoadContext> Worklist;
  Worklist.push({ArgInParamAS, 0});

  while (!Worklist.empty()) {
    LoadContext Ctx = Worklist.front();
    Worklist.pop();

    for (User *CurUser : Ctx.InitialVal->users()) {
      if (auto *I = dyn_cast<LoadInst>(CurUser))
        Loads.push_back({I, Ctx.Offset});
      else if (isa<BitCastInst>(CurUser) || isa<AddrSpaceCastInst>(CurUser))
        Worklist.push({cast<Instruction>(CurUser), Ctx.Offset});
      else if (auto *I = dyn_cast<GetElementPtrInst>(CurUser)) {
        APInt OffsetAccumulated =
            APInt::getZero(DL.getIndexSizeInBits(ADDRESS_SPACE_PARAM));

        if (!I->accumulateConstantOffset(DL, OffsetAccumulated))
          continue;

        uint64_t OffsetLimit = -1;
        uint64_t Offset = OffsetAccumulated.getLimitedValue(OffsetLimit);
        assert(Offset != OffsetLimit && "Expect Offset less than UINT64_MAX");

        Worklist.push({I, Ctx.Offset + Offset});
      }
    }
  }

  for (Load &CurLoad : Loads) {
    Align NewLoadAlign(std::gcd(NewArgAlign.value(), CurLoad.Offset));
    Align CurLoadAlign = CurLoad.Inst->getAlign();
    CurLoad.Inst->setAlignment(std::max(NewLoadAlign, CurLoadAlign));
  }
}

// Create a call to the nvvm_internal_addrspace_wrap intrinsic and set the
// alignment of the return value based on the alignment of the argument.
static CallInst *createNVVMInternalAddrspaceWrap(IRBuilder<> &IRB,
                                                 Argument &Arg) {
  CallInst *ArgInParam =
      IRB.CreateIntrinsic(Intrinsic::nvvm_internal_addrspace_wrap,
                          {IRB.getPtrTy(ADDRESS_SPACE_PARAM), Arg.getType()},
                          &Arg, {}, Arg.getName() + ".param");

  if (MaybeAlign ParamAlign = Arg.getParamAlign())
    ArgInParam->addRetAttr(
        Attribute::getWithAlignment(ArgInParam->getContext(), *ParamAlign));

  return ArgInParam;
}

namespace {
struct ArgUseChecker : PtrUseVisitor<ArgUseChecker> {
  using Base = PtrUseVisitor<ArgUseChecker>;

  bool IsGridConstant;
  // Set of phi/select instructions using the Arg
  SmallPtrSet<Instruction *, 4> Conditionals;

  ArgUseChecker(const DataLayout &DL, bool IsGridConstant)
      : PtrUseVisitor(DL), IsGridConstant(IsGridConstant) {}

  PtrInfo visitArgPtr(Argument &A) {
    assert(A.getType()->isPointerTy());
    IntegerType *IntIdxTy = cast<IntegerType>(DL.getIndexType(A.getType()));
    IsOffsetKnown = false;
    Offset = APInt(IntIdxTy->getBitWidth(), 0);
    PI.reset();
    Conditionals.clear();

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
      if (isa<PHINode>(I) || isa<SelectInst>(I))
        Conditionals.insert(I);
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
    // Writes to the pointer are UB w/ __grid_constant__, but do not force a
    // copy.
    if (!IsGridConstant)
      return PI.setAborted(&SI);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // ASC to param space are no-ops and do not need a copy
    if (ASC.getDestAddressSpace() != ADDRESS_SPACE_PARAM)
      return PI.setEscapedAndAborted(&ASC);
    Base::visitAddrSpaceCastInst(ASC);
  }

  void visitPtrToIntInst(PtrToIntInst &I) {
    if (IsGridConstant)
      return;
    Base::visitPtrToIntInst(I);
  }
  void visitPHINodeOrSelectInst(Instruction &I) {
    assert(isa<PHINode>(I) || isa<SelectInst>(I));
  }
  // PHI and select just pass through the pointers.
  void visitPHINode(PHINode &PN) { enqueueUsers(PN); }
  void visitSelectInst(SelectInst &SI) { enqueueUsers(SI); }

  void visitMemTransferInst(MemTransferInst &II) {
    if (*U == II.getRawDest() && !IsGridConstant)
      PI.setAborted(&II);
    // memcpy/memmove are OK when the pointer is source. We can convert them to
    // AS-specific memcpy.
  }

  void visitMemSetInst(MemSetInst &II) {
    if (!IsGridConstant)
      PI.setAborted(&II);
  }
}; // struct ArgUseChecker

void copyByValParam(Function &F, Argument &Arg) {
  LLVM_DEBUG(dbgs() << "Creating a local copy of " << Arg << "\n");
  // Otherwise we have to create a temporary copy.
  BasicBlock::iterator FirstInst = F.getEntryBlock().begin();
  Type *StructType = Arg.getParamByValType();
  const DataLayout &DL = F.getDataLayout();
  IRBuilder<> IRB(&*FirstInst);
  AllocaInst *AllocA = IRB.CreateAlloca(StructType, nullptr, Arg.getName());
  // Set the alignment to alignment of the byval parameter. This is because,
  // later load/stores assume that alignment, and we are going to replace
  // the use of the byval parameter with this alloca instruction.
  AllocA->setAlignment(
      Arg.getParamAlign().value_or(DL.getPrefTypeAlign(StructType)));
  Arg.replaceAllUsesWith(AllocA);

  CallInst *ArgInParam = createNVVMInternalAddrspaceWrap(IRB, Arg);

  // Be sure to propagate alignment to this load; LLVM doesn't know that NVPTX
  // addrspacecast preserves alignment.  Since params are constant, this load
  // is definitely not volatile.
  const auto ArgSize = *AllocA->getAllocationSize(DL);
  IRB.CreateMemCpy(AllocA, AllocA->getAlign(), ArgInParam, AllocA->getAlign(),
                   ArgSize);
}
} // namespace

static void handleByValParam(const NVPTXTargetMachine &TM, Argument *Arg) {
  Function *Func = Arg->getParent();
  assert(isKernelFunction(*Func));
  const bool HasCvtaParam = TM.getSubtargetImpl(*Func)->hasCvtaParam();
  const bool IsGridConstant = HasCvtaParam && isParamGridConstant(*Arg);
  const DataLayout &DL = Func->getDataLayout();
  BasicBlock::iterator FirstInst = Func->getEntryBlock().begin();
  [[maybe_unused]] Type *StructType = Arg->getParamByValType();
  assert(StructType && "Missing byval type");

  ArgUseChecker AUC(DL, IsGridConstant);
  ArgUseChecker::PtrInfo PI = AUC.visitArgPtr(*Arg);
  bool ArgUseIsReadOnly = !(PI.isEscaped() || PI.isAborted());
  // Easy case, accessing parameter directly is fine.
  if (ArgUseIsReadOnly && AUC.Conditionals.empty()) {
    // Convert all loads and intermediate operations to use parameter AS and
    // skip creation of a local copy of the argument.
    SmallVector<Use *, 16> UsesToUpdate(llvm::make_pointer_range(Arg->uses()));

    IRBuilder<> IRB(&*FirstInst);
    CallInst *ArgInParamAS = createNVVMInternalAddrspaceWrap(IRB, *Arg);

    for (Use *U : UsesToUpdate)
      convertToParamAS(U, ArgInParamAS, HasCvtaParam, IsGridConstant);
    LLVM_DEBUG(dbgs() << "No need to copy or cast " << *Arg << "\n");

    const auto *TLI =
        cast<NVPTXTargetLowering>(TM.getSubtargetImpl()->getTargetLowering());

    adjustByValArgAlignment(Arg, ArgInParamAS, TLI);

    return;
  }

  // We can't access byval arg directly and need a pointer. on sm_70+ we have
  // ability to take a pointer to the argument without making a local copy.
  // However, we're still not allowed to write to it. If the user specified
  // `__grid_constant__` for the argument, we'll consider escaped pointer as
  // read-only.
  if (IsGridConstant || (HasCvtaParam && ArgUseIsReadOnly)) {
    LLVM_DEBUG(dbgs() << "Using non-copy pointer to " << *Arg << "\n");
    // Replace all argument pointer uses (which might include a device function
    // call) with a cast to the generic address space using cvta.param
    // instruction, which avoids a local copy.
    IRBuilder<> IRB(&Func->getEntryBlock().front());

    // Cast argument to param address space. Because the backend will emit the
    // argument already in the param address space, we need to use the noop
    // intrinsic, this had the added benefit of preventing other optimizations
    // from folding away this pair of addrspacecasts.
    auto *ParamSpaceArg = createNVVMInternalAddrspaceWrap(IRB, *Arg);

    // Cast param address to generic address space.
    Value *GenericArg = IRB.CreateAddrSpaceCast(
        ParamSpaceArg, IRB.getPtrTy(ADDRESS_SPACE_GENERIC),
        Arg->getName() + ".gen");

    Arg->replaceAllUsesWith(GenericArg);

    // Do not replace Arg in the cast to param space
    ParamSpaceArg->setOperand(0, Arg);
  } else
    copyByValParam(*Func, *Arg);
}

static void markPointerAsAS(Value *Ptr, const unsigned AS) {
  if (Ptr->getType()->getPointerAddressSpace() != ADDRESS_SPACE_GENERIC)
    return;

  // Deciding where to emit the addrspacecast pair.
  BasicBlock::iterator InsertPt;
  if (Argument *Arg = dyn_cast<Argument>(Ptr)) {
    // Insert at the functon entry if Ptr is an argument.
    InsertPt = Arg->getParent()->getEntryBlock().begin();
  } else {
    // Insert right after Ptr if Ptr is an instruction.
    InsertPt = ++cast<Instruction>(Ptr)->getIterator();
    assert(InsertPt != InsertPt->getParent()->end() &&
           "We don't call this function with Ptr being a terminator.");
  }

  Instruction *PtrInGlobal = new AddrSpaceCastInst(
      Ptr, PointerType::get(Ptr->getContext(), AS), Ptr->getName(), InsertPt);
  Value *PtrInGeneric = new AddrSpaceCastInst(PtrInGlobal, Ptr->getType(),
                                              Ptr->getName(), InsertPt);
  // Replace with PtrInGeneric all uses of Ptr except PtrInGlobal.
  Ptr->replaceAllUsesWith(PtrInGeneric);
  PtrInGlobal->setOperand(0, Ptr);
}

static void markPointerAsGlobal(Value *Ptr) {
  markPointerAsAS(Ptr, ADDRESS_SPACE_GLOBAL);
}

// =============================================================================
// Main function for this pass.
// =============================================================================
static bool runOnKernelFunction(const NVPTXTargetMachine &TM, Function &F) {
  // Copying of byval aggregates + SROA may result in pointers being loaded as
  // integers, followed by intotoptr. We may want to mark those as global, too,
  // but only if the loaded integer is used exclusively for conversion to a
  // pointer with inttoptr.
  auto HandleIntToPtr = [](Value &V) {
    if (llvm::all_of(V.users(), [](User *U) { return isa<IntToPtrInst>(U); })) {
      SmallVector<User *, 16> UsersToUpdate(V.users());
      for (User *U : UsersToUpdate)
        markPointerAsGlobal(U);
    }
  };
  if (TM.getDrvInterface() == NVPTX::CUDA) {
    // Mark pointers in byval structs as global.
    for (auto &B : F) {
      for (auto &I : B) {
        if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getType()->isPointerTy() || LI->getType()->isIntegerTy()) {
            Value *UO = getUnderlyingObject(LI->getPointerOperand());
            if (Argument *Arg = dyn_cast<Argument>(UO)) {
              if (Arg->hasByValAttr()) {
                // LI is a load from a pointer within a byval kernel parameter.
                if (LI->getType()->isPointerTy())
                  markPointerAsGlobal(LI);
                else
                  HandleIntToPtr(*LI);
              }
            }
          }
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Lowering kernel args of " << F.getName() << "\n");
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() && Arg.hasByValAttr()) {
      handleByValParam(TM, &Arg);
    } else if (Arg.getType()->isIntegerTy() &&
               TM.getDrvInterface() == NVPTX::CUDA) {
      HandleIntToPtr(Arg);
    }
  }
  return true;
}

// Device functions only need to copy byval args into local memory.
static bool runOnDeviceFunction(const NVPTXTargetMachine &TM, Function &F) {
  LLVM_DEBUG(dbgs() << "Lowering function args of " << F.getName() << "\n");

  const auto *TLI =
      cast<NVPTXTargetLowering>(TM.getSubtargetImpl()->getTargetLowering());

  for (Argument &Arg : F.args())
    if (Arg.getType()->isPointerTy() && Arg.hasByValAttr())
      adjustByValArgAlignment(&Arg, &Arg, TLI);

  return true;
}

static bool processFunction(Function &F, NVPTXTargetMachine &TM) {
  return isKernelFunction(F) ? runOnKernelFunction(TM, F)
                             : runOnDeviceFunction(TM, F);
}

bool NVPTXLowerArgsLegacyPass::runOnFunction(Function &F) {
  auto &TM = getAnalysis<TargetPassConfig>().getTM<NVPTXTargetMachine>();
  return processFunction(F, TM);
}
FunctionPass *llvm::createNVPTXLowerArgsPass() {
  return new NVPTXLowerArgsLegacyPass();
}

static bool copyFunctionByValArgs(Function &F) {
  LLVM_DEBUG(dbgs() << "Creating a copy of byval args of " << F.getName()
                    << "\n");
  bool Changed = false;
  if (isKernelFunction(F)) {
    for (Argument &Arg : F.args())
      if (Arg.getType()->isPointerTy() && Arg.hasByValAttr() &&
          !isParamGridConstant(Arg)) {
        copyByValParam(F, Arg);
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

PreservedAnalyses NVPTXLowerArgsPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  auto &NTM = static_cast<NVPTXTargetMachine &>(TM);
  bool Changed = processFunction(F, NTM);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
