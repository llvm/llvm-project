//===-- InlineAsmPrepare - Prepare inline asm for code generation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers callbrs and inline asm in LLVM IR in order to assist
// SelectionDAG's codegen.
//
// CallBrInst:
//
//   - Assists in inserting register copies for the output values of a callbr
//     along the edges leading to the indirect target blocks. Though the output
//     SSA value is defined by the callbr instruction itself in the IR
//     representation, the value cannot be copied to the appropriate virtual
//     registers prior to jumping to an indirect label, since the jump occurs
//     within the user-provided assembly blob.
//
//     Instead, those copies must occur separately at the beginning of each
//     indirect target. That requires that we create a separate SSA definition
//     in each of them (via llvm.callbr.landingpad), and may require splitting
//     critical edges so we have a location to place the intrinsic. Finally, we
//     remap users of the original callbr output SSA value to instead point to
//     the appropriate llvm.callbr.landingpad value.
//
//     Ideally, this could be done inside SelectionDAG, or in the
//     MachineInstruction representation, without the use of an IR-level
//     intrinsic.  But, within the current framework, it’s simpler to implement
//     as an IR pass.  (If support for callbr in GlobalISel is implemented,
//     it’s worth considering whether this is still required.)
//
// InlineAsm:
//
//   - Prepares inline assembly for code generation with the fast register
//     allocator. In particular, it defaults "rm" (register-or-memory) to
//     prefer the "m" constraints (the front-end opts for the "r" constraint),
//     simplifying register allocation by forcing operands to memory locations.
//     The other register allocators are equipped to handle folding registers
//     already, so don't need to change the default.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InlineAsmPrepare.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "inline-asm-prepare"

namespace {

class InlineAsmPrepare : public FunctionPass {
public:
  InlineAsmPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
  bool runOnFunction(Function &F) override;

  static char ID;
};

char InlineAsmPrepare::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(InlineAsmPrepare, DEBUG_TYPE, "Prepare inline asm insts",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(InlineAsmPrepare, DEBUG_TYPE, "Prepare inline asm insts",
                    false, false)

FunctionPass *llvm::createInlineAsmPreparePass() {
  return new InlineAsmPrepare();
}

//===----------------------------------------------------------------------===//
//                     Process InlineAsm instructions
//===----------------------------------------------------------------------===//

/// The inline asm constraint allows both register and memory.
static bool IsRegMemConstraint(StringRef Constraint) {
  return Constraint.size() == 2 && (Constraint == "rm" || Constraint == "mr");
}

/// Tag "rm" output constraints with '*' to signify that they default to a
/// memory location.
static std::pair<std::string, bool>
ConvertConstraintsToMemory(StringRef ConstraintStr) {
  auto I = ConstraintStr.begin(), E = ConstraintStr.end();
  std::string Out;
  raw_string_ostream O(Out);
  bool HasRegMem = false;

  while (I != E) {
    bool IsOutput = false;
    bool HasIndirect = false;
    if (*I == '=') {
      O << *I;
      IsOutput = true;
      ++I;
      if (I == E)
        return {};
    }
    if (*I == '*') {
      O << '*';
      HasIndirect = true;
      ++I;
      if (I == E)
        return {};
    }
    if (*I == '+') {
      O << '+';
      IsOutput = true;
      ++I;
      if (I == E)
        return {};
    }

    auto Comma = std::find(I, E, ',');
    std::string Sub(I, Comma);
    if (IsRegMemConstraint(Sub)) {
      HasRegMem = true;
      if (IsOutput && !HasIndirect)
        O << '*';
    }

    O << Sub;

    if (Comma == E)
      break;

    O << ',';
    I = Comma + 1;
  }

  return {Out, HasRegMem};
}

/// Build a map of tied constraints. TiedOutput[i] = j means Constraint i is an
/// input tied to output constraint j.
static void
BuildTiedConstraintMap(const InlineAsm::ConstraintInfoVector &Constraints,
                       SmallVectorImpl<int> &TiedOutput) {
  for (unsigned I = 0, E = Constraints.size(); I != E; ++I) {
    const InlineAsm::ConstraintInfo &C = Constraints[I];
    if (C.Type == InlineAsm::isOutput && C.hasMatchingInput()) {
      int InputIdx = C.MatchingInput;
      if (InputIdx >= 0 && InputIdx < (int)Constraints.size())
        TiedOutput[InputIdx] = I;
    }

    if (C.Type == InlineAsm::isInput && C.hasMatchingInput()) {
      int OutputIdx = C.MatchingInput;
      if (OutputIdx >= 0 && OutputIdx < (int)Constraints.size())
        TiedOutput[I] = OutputIdx;
    }
  }
}

/// Process an output constraint, creating allocas for converted constraints.
static void ProcessOutputConstraint(
    const InlineAsm::ConstraintInfo &C, Type *RetTy, unsigned OutputIdx,
    IRBuilder<> &EntryBuilder, SmallVectorImpl<Value *> &NewArgs,
    SmallVectorImpl<Type *> &NewArgTypes, SmallVectorImpl<Type *> &NewRetTypes,
    SmallVectorImpl<std::pair<unsigned, Type *>> &ElementTypeAttrs,
    SmallVectorImpl<AllocaInst *> &OutputAllocas, unsigned ConstraintIdx) {
  Type *SlotTy = RetTy;
  if (StructType *ST = dyn_cast<StructType>(RetTy))
    SlotTy = ST->getElementType(OutputIdx);

  if (C.hasRegMemConstraints()) {
    // Converted to memory constraint. Create alloca and pass pointer as
    // argument.
    AllocaInst *Slot = EntryBuilder.CreateAlloca(SlotTy, nullptr, "asm_mem");
    NewArgs.push_back(Slot);
    NewArgTypes.push_back(Slot->getType());
    ElementTypeAttrs.push_back({NewArgs.size() - 1, SlotTy});
    OutputAllocas[ConstraintIdx] = Slot;
    // No return value for this output since it's now an out-parameter.
  } else {
    // Unchanged, still an output return value.
    NewRetTypes.push_back(SlotTy);
  }
}

/// Process an input constraint, handling tied constraints and conversions.
static void ProcessInputConstraint(const InlineAsm::ConstraintInfo &C,
                                   Value *ArgVal, ArrayRef<int> TiedOutput,
                                   ArrayRef<AllocaInst *> OutputAllocas,
                                   unsigned ConstraintIdx, IRBuilder<> &Builder,
                                   IRBuilder<> &EntryBuilder,
                                   SmallVectorImpl<Value *> &NewArgs,
                                   SmallVectorImpl<Type *> &NewArgTypes) {
  Type *ArgTy = ArgVal->getType();

  if (TiedOutput[ConstraintIdx] != -1) {
    int MatchIdx = TiedOutput[ConstraintIdx];
    if (AllocaInst *Slot = OutputAllocas[MatchIdx]) {
      // The matched output was converted to memory. Store this input into the
      // alloca.
      Builder.CreateStore(ArgVal, Slot);

      // Pass the alloca pointer as the argument, instead of ArgVal. This
      // ensures the tied "0" constraint matches the "*m" output.
      NewArgs.push_back(Slot);
      NewArgTypes.push_back(Slot->getType());
      return;
    }
  }

  if (C.hasRegMemConstraints()) {
    // Converted to memory constraint. Create alloca, store input, pass pointer
    // as argument.
    AllocaInst *Slot = EntryBuilder.CreateAlloca(ArgTy, nullptr, "asm_mem");
    Builder.CreateStore(ArgVal, Slot);
    NewArgs.push_back(Slot);
    NewArgTypes.push_back(Slot->getType());
  } else {
    // Unchanged
    NewArgs.push_back(ArgVal);
    NewArgTypes.push_back(ArgTy);
  }
}

/// Build the return type from the collected return types.
static Type *BuildReturnType(ArrayRef<Type *> NewRetTypes,
                             LLVMContext &Context) {
  if (NewRetTypes.empty())
    return Type::getVoidTy(Context);

  if (NewRetTypes.size() == 1)
    return NewRetTypes[0];

  return StructType::get(Context, NewRetTypes);
}

/// Create the new inline assembly call with converted constraints.
static CallInst *CreateNewInlineAsm(
    InlineAsm *IA, const std::string &NewConstraintStr, Type *NewRetTy,
    const SmallVectorImpl<Type *> &NewArgTypes,
    const SmallVectorImpl<Value *> &NewArgs,
    const SmallVectorImpl<std::pair<unsigned, Type *>> &ElementTypeAttrs,
    CallBase *CB, IRBuilder<> &Builder, LLVMContext &Context) {
  FunctionType *NewFTy = FunctionType::get(NewRetTy, NewArgTypes, false);
  InlineAsm *NewIA = InlineAsm::get(
      NewFTy, IA->getAsmString(), NewConstraintStr, IA->hasSideEffects(),
      IA->isAlignStack(), IA->getDialect(), IA->canThrow());

  CallInst *NewCall = Builder.CreateCall(NewFTy, NewIA, NewArgs);
  NewCall->setCallingConv(CB->getCallingConv());
  NewCall->setAttributes(CB->getAttributes());
  NewCall->setDebugLoc(CB->getDebugLoc());

  for (const std::pair<unsigned, Type *> &Item : ElementTypeAttrs)
    NewCall->addParamAttr(
        Item.first,
        Attribute::get(Context, Attribute::ElementType, Item.second));

  return NewCall;
}

/// Reconstruct the return value from the new call and allocas.
static Value *
ReconstructReturnValue(Type *RetTy, CallInst *NewCall,
                       const InlineAsm::ConstraintInfoVector &Constraints,
                       const SmallVectorImpl<AllocaInst *> &OutputAllocas,
                       const SmallVectorImpl<Type *> &NewRetTypes,
                       IRBuilder<> &Builder) {
  if (RetTy->isVoidTy())
    return nullptr;

  if (isa<StructType>(RetTy)) {
    // Multiple outputs. Reconstruct the struct.
    Value *Res = PoisonValue::get(RetTy);
    unsigned NewRetIdx = 0;
    unsigned OriginalOutIdx = 0;

    for (unsigned I = 0, E = Constraints.size(); I != E; ++I) {
      if (Constraints[I].Type != InlineAsm::isOutput)
        continue;

      Value *Val = nullptr;
      if (AllocaInst *Slot = OutputAllocas[I]) {
        // Converted to memory. Load from alloca.
        Val = Builder.CreateLoad(Slot->getAllocatedType(), Slot);
      } else {
        // Not converted. Extract from NewCall return.
        if (NewRetTypes.size() == 1) {
          Val = NewCall;
        } else {
          Val = Builder.CreateExtractValue(NewCall, NewRetIdx);
        }
        NewRetIdx++;
      }

      Res = Builder.CreateInsertValue(Res, Val, OriginalOutIdx++);
    }

    return Res;
  }

  // Single output.
  // Find the output constraint (should be the first one).
  unsigned OutConstraintIdx = 0;
  for (unsigned I = 0; I < Constraints.size(); ++I) {
    if (Constraints[I].Type == InlineAsm::isOutput) {
      OutConstraintIdx = I;
      break;
    }
  }

  if (AllocaInst *Slot = OutputAllocas[OutConstraintIdx])
    return Builder.CreateLoad(Slot->getAllocatedType(), Slot);

  return NewCall;
}

static bool ProcessInlineAsm(Function &F, CallBase *CB) {
  InlineAsm *IA = cast<InlineAsm>(CB->getCalledOperand());
  const InlineAsm::ConstraintInfoVector &Constraints = IA->ParseConstraints();

  auto [NewConstraintStr, HasRegMem] =
      ConvertConstraintsToMemory(IA->getConstraintString());
  if (!HasRegMem)
    return false;

  IRBuilder<> Builder(CB);
  IRBuilder<> EntryBuilder(&F.getEntryBlock(), F.getEntryBlock().begin());

  // Collect new arguments and return types.
  SmallVector<Value *, 8> NewArgs;
  SmallVector<Type *, 8> NewArgTypes;
  SmallVector<Type *, 2> NewRetTypes;
  SmallVector<std::pair<unsigned, Type *>, 8> ElementTypeAttrs;

  // Track allocas created for converted outputs. Indexed by position in the
  // flat Constraints list (not by output index), so that both
  // ProcessOutputConstraint and ReconstructReturnValue can look up entries
  // using the same constraint index.
  SmallVector<AllocaInst *, 8> OutputAllocas(Constraints.size(), nullptr);

  // Build tied constraint map.
  SmallVector<int, 8> TiedOutput(Constraints.size(), -1);
  BuildTiedConstraintMap(Constraints, TiedOutput);

  // Process constraints.
  unsigned ArgNo = 0;
  unsigned OutputIdx = 0;
  for (unsigned I = 0, E = Constraints.size(); I != E; ++I) {
    const InlineAsm::ConstraintInfo &C = Constraints[I];

    if (C.Type == InlineAsm::isOutput) {
      if (C.isIndirect) {
        // Indirect output takes a pointer argument from the original call.
        // Pass it through to the new call.
        Value *ArgVal = CB->getArgOperand(ArgNo);
        NewArgs.push_back(ArgVal);
        NewArgTypes.push_back(ArgVal->getType());
        // Preserve element type attribute if present.
        if (auto *Ty = CB->getParamElementType(ArgNo))
          ElementTypeAttrs.push_back({NewArgs.size() - 1, Ty});
        ArgNo++;
      } else {
        ProcessOutputConstraint(C, CB->getType(), OutputIdx, EntryBuilder,
                                NewArgs, NewArgTypes, NewRetTypes,
                                ElementTypeAttrs, OutputAllocas, I);
        OutputIdx++;
      }
    } else if (C.Type == InlineAsm::isInput) {
      Value *ArgVal = CB->getArgOperand(ArgNo);
      ProcessInputConstraint(C, ArgVal, TiedOutput, OutputAllocas, I, Builder,
                             EntryBuilder, NewArgs, NewArgTypes);
      ArgNo++;
    }
  }

  // Build the new return type.
  Type *NewRetTy = BuildReturnType(NewRetTypes, F.getContext());

  // Create the new inline assembly call.
  CallInst *NewCall =
      CreateNewInlineAsm(IA, NewConstraintStr, NewRetTy, NewArgTypes, NewArgs,
                         ElementTypeAttrs, CB, Builder, F.getContext());

  // Reconstruct the return value and update users.
  if (!CB->use_empty()) {
    if (Value *Replacement =
            ReconstructReturnValue(CB->getType(), NewCall, Constraints,
                                   OutputAllocas, NewRetTypes, Builder))
      CB->replaceAllUsesWith(Replacement);
  }

  CB->eraseFromParent();
  return true;
}

//===----------------------------------------------------------------------===//
//                           Process CallBrInsts
//===----------------------------------------------------------------------===//

/// The Use is in the same BasicBlock as the intrinsic call.
static bool IsInSameBasicBlock(const Use &U, const BasicBlock *BB) {
  const auto *I = dyn_cast<Instruction>(U.getUser());
  return I && I->getParent() == BB;
}

#ifndef NDEBUG
static void PrintDebugDomInfo(const DominatorTree &DT, const Use &U,
                              const BasicBlock *BB, bool IsDefaultDest) {
  if (isa<Instruction>(U.getUser()))
    LLVM_DEBUG(dbgs() << "Use: " << *U.getUser() << ", in block "
                      << cast<Instruction>(U.getUser())->getParent()->getName()
                      << ", is " << (DT.dominates(BB, U) ? "" : "NOT ")
                      << "dominated by " << BB->getName() << " ("
                      << (IsDefaultDest ? "in" : "") << "direct)\n");
}
#endif

static void UpdateSSA(DominatorTree &DT, CallBrInst *CBR, CallInst *Intrinsic,
                      SSAUpdater &SSAUpdate) {
  SmallPtrSet<Use *, 4> Visited;

  BasicBlock *DefaultDest = CBR->getDefaultDest();
  BasicBlock *LandingPad = Intrinsic->getParent();
  SmallVector<Use *, 4> Uses(make_pointer_range(CBR->uses()));

  for (Use *U : Uses) {
    if (!Visited.insert(U).second)
      continue;

#ifndef NDEBUG
    PrintDebugDomInfo(DT, *U, LandingPad, /*IsDefaultDest*/ false);
    PrintDebugDomInfo(DT, *U, DefaultDest, /*IsDefaultDest*/ true);
#endif

    // Don't rewrite the use in the newly inserted intrinsic.
    if (const auto *II = dyn_cast<IntrinsicInst>(U->getUser()))
      if (II->getIntrinsicID() == Intrinsic::callbr_landingpad)
        continue;

    // If the Use is in the same BasicBlock as the Intrinsic call, replace
    // the Use with the value of the Intrinsic call.
    if (IsInSameBasicBlock(*U, LandingPad)) {
      U->set(Intrinsic);
      continue;
    }

    // If the Use is dominated by the default dest, do not touch it.
    if (DT.dominates(DefaultDest, *U))
      continue;

    SSAUpdate.RewriteUse(*U);
  }
}

static bool SplitCriticalEdges(CallBrInst *CBR, DominatorTree *DT) {
  bool Changed = false;

  CriticalEdgeSplittingOptions Options(DT);
  Options.setMergeIdenticalEdges();

  // The indirect destination might be duplicated between another parameter...
  //
  //   %0 = callbr ... [label %x, label %x]
  //
  // ...hence MergeIdenticalEdges and AllowIndentical edges, but we don't need
  // to split the default destination if it's duplicated between an indirect
  // destination...
  //
  //   %1 = callbr ... to label %x [label %x]
  //
  // ...hence starting at 1 and checking against successor 0 (aka the default
  // destination).
  for (unsigned i = 1, e = CBR->getNumSuccessors(); i != e; ++i)
    if (CBR->getSuccessor(i) == CBR->getSuccessor(0) ||
        isCriticalEdge(CBR, i, /*AllowIdenticalEdges*/ true))
      if (SplitKnownCriticalEdge(CBR, i, Options))
        Changed = true;

  return Changed;
}

static bool InsertIntrinsicCalls(CallBrInst *CBR, DominatorTree &DT) {
  bool Changed = false;
  SmallPtrSet<const BasicBlock *, 4> Visited;
  IRBuilder<> Builder(CBR->getContext());

  if (!CBR->getNumIndirectDests())
    return false;

  SSAUpdater SSAUpdate;
  SSAUpdate.Initialize(CBR->getType(), CBR->getName());
  SSAUpdate.AddAvailableValue(CBR->getParent(), CBR);
  SSAUpdate.AddAvailableValue(CBR->getDefaultDest(), CBR);

  for (BasicBlock *IndDest : CBR->getIndirectDests()) {
    if (!Visited.insert(IndDest).second)
      continue;

    Builder.SetInsertPoint(&*IndDest->begin());
    CallInst *Intrinsic = Builder.CreateIntrinsic(
        CBR->getType(), Intrinsic::callbr_landingpad, {CBR});
    SSAUpdate.AddAvailableValue(IndDest, Intrinsic);
    UpdateSSA(DT, CBR, Intrinsic, SSAUpdate);
    Changed = true;
  }

  return Changed;
}

static bool ProcessCallBrInst(Function &F, CallBrInst *CBR, DominatorTree *DT) {
  bool Changed = false;

  Changed |= SplitCriticalEdges(CBR, DT);
  Changed |= InsertIntrinsicCalls(CBR, *DT);

  return Changed;
}

static bool runImpl(Function &F, ArrayRef<CallBase *> IAs, DominatorTree *DT) {
  bool Changed = false;

  for (CallBase *CB : IAs)
    if (auto *CBR = dyn_cast<CallBrInst>(CB))
      Changed |= ProcessCallBrInst(F, CBR, DT);
    else
      Changed |= ProcessInlineAsm(F, CB);

  return Changed;
}

/// Find all inline assembly calls that need preparation. This always collects
/// CallBrInsts (which need SSA fixups), and at -O0 also collects regular
/// inline asm calls (which need "rm" to "m" constraint conversion for the fast
/// register allocator).
static SmallVector<CallBase *, 4>
FindInlineAsmCandidates(Function &F, const TargetMachine *TM) {
  bool isOptLevelNone = TM->getOptLevel() == CodeGenOptLevel::None;
  SmallVector<CallBase *, 4> InlineAsms;

  for (BasicBlock &BB : F) {
    if (auto *CBR = dyn_cast<CallBrInst>(BB.getTerminator())) {
      if (!CBR->getType()->isVoidTy() && !CBR->use_empty())
        InlineAsms.push_back(CBR);
      continue;
    }

    if (isOptLevelNone)
      // Only inline assembly compiled at '-O0' (i.e. uses the fast register
      // allocator) needs to be processed.
      for (Instruction &I : BB)
        if (CallBase *CB = dyn_cast<CallBase>(&I); CB && CB->isInlineAsm())
          InlineAsms.push_back(CB);
  }

  return InlineAsms;
}

bool InlineAsmPrepare::runOnFunction(Function &F) {
  const auto *TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  SmallVector<CallBase *, 4> IAs = FindInlineAsmCandidates(F, TM);
  if (IAs.empty())
    return false;

  // It's highly likely that most programs do not contain CallBrInsts. Follow a
  // similar pattern from SafeStackLegacyPass::runOnFunction to reuse previous
  // domtree analysis if available, otherwise compute it lazily. This avoids
  // forcing Dominator Tree Construction at -O0 for programs that likely do not
  // contain CallBrInsts. It does pessimize programs with callbr at higher
  // optimization levels, as the DominatorTree created here is not reused by
  // subsequent passes.
  DominatorTree *DT;
  std::optional<DominatorTree> LazilyComputedDomTree;
  if (auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
    DT = &DTWP->getDomTree();
  else {
    LazilyComputedDomTree.emplace(F);
    DT = &*LazilyComputedDomTree;
  }

  return runImpl(F, IAs, DT);
}

PreservedAnalyses InlineAsmPreparePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  SmallVector<CallBase *, 4> IAs = FindInlineAsmCandidates(F, TM);
  if (IAs.empty())
    return PreservedAnalyses::all();

  DominatorTree *DT = &FAM.getResult<DominatorTreeAnalysis>(F);

  if (runImpl(F, IAs, DT)) {
    PreservedAnalyses PA;
    PA.preserve<DominatorTreeAnalysis>();
    return PA;
  }

  return PreservedAnalyses::all();
}
