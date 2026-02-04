//===-- InlineAsmPrepare - Prepare inline asm for code generation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass prepares inline assembly for code generation with the fast register
// allocator---e.g., by converting "rm" (register-or-memory) constraints to "m"
// (memory-only) constraints, simplifying register allocation by forcing
// operands to memory locations, avoiding the complexity of handling dual
// register/memory options. The other register allocators are equipped to
// handle folding registers all ready.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InlineAsmPrepare.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "inline-asm-prepare"

namespace {

class InlineAsmPrepare : public FunctionPass {
  InlineAsmPrepare(InlineAsmPrepare &) = delete;

public:
  InlineAsmPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
  bool runOnFunction(Function &F) override;

  static char ID;
};

char InlineAsmPrepare::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(InlineAsmPrepare, DEBUG_TYPE,
                "Prepare inline asm insts for fast register allocation", false,
                false)
FunctionPass *llvm::createInlineAsmPass() { return new InlineAsmPrepare(); }

/// Find all inline assembly calls in the given function.
static SmallVector<CallBase *, 4> findInlineAsms(Function &F) {
  SmallVector<CallBase *, 4> InlineAsms;

  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (CallBase *CB = dyn_cast<CallBase>(&I); CB && CB->isInlineAsm())
        InlineAsms.push_back(CB);

  return InlineAsms;
}

static bool isRegMemConstraint(StringRef Constraint) {
  return Constraint.size() == 2 && (Constraint == "rm" || Constraint == "mr");
}

/// Convert instances of the "rm" constraints into "m".
static std::pair<std::string, bool>
convertConstraintsToMemory(StringRef ConstraintStr) {
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
    }
    if (*I == '*') {
      O << '*';
      HasIndirect = true;
      ++I;
    }
    if (*I == '+') {
      O << '+';
      IsOutput = true;
      ++I;
    }

    auto Comma = std::find(I, E, ',');
    std::string Sub(I, Comma);
    if (isRegMemConstraint(Sub)) {
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

  return std::make_pair(Out, HasRegMem);
}

namespace {

/// Build a map of tied constraints.
/// TiedOutput[i] = j means Constraint i is an Input tied to Output Constraint
/// j.
static void
buildTiedConstraintMap(const InlineAsm::ConstraintInfoVector &Constraints,
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
static void processOutputConstraint(
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
static void processInputConstraint(
    const InlineAsm::ConstraintInfo &C, Value *ArgVal,
    const SmallVectorImpl<int> &TiedOutput,
    const SmallVectorImpl<AllocaInst *> &OutputAllocas, unsigned ConstraintIdx,
    IRBuilder<> &Builder, IRBuilder<> &EntryBuilder,
    SmallVectorImpl<Value *> &NewArgs, SmallVectorImpl<Type *> &NewArgTypes) {
  Type *ArgTy = ArgVal->getType();
  bool Handled = false;

  if (TiedOutput[ConstraintIdx] != -1) {
    int MatchIdx = TiedOutput[ConstraintIdx];
    if (AllocaInst *Slot = OutputAllocas[MatchIdx]) {
      // The matched output was converted to memory.
      // Store this input into the alloca.
      Builder.CreateStore(ArgVal, Slot);
      // Pass the alloca pointer as the argument, instead of ArgVal.
      // This ensures the tied "0" constraint matches the "*m" output.
      NewArgs.push_back(Slot);
      NewArgTypes.push_back(Slot->getType());
      Handled = true;
    }
  }

  if (!Handled) {
    if (C.hasRegMemConstraints()) {
      // Converted to memory constraint.
      // Create alloca, store input, pass pointer as argument.
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
}

/// Build the return type from the collected return types.
static Type *buildReturnType(const SmallVectorImpl<Type *> &NewRetTypes,
                             LLVMContext &Context) {
  if (NewRetTypes.empty())
    return Type::getVoidTy(Context);
  if (NewRetTypes.size() == 1)
    return NewRetTypes[0];
  return StructType::get(Context, NewRetTypes);
}

/// Create the new inline assembly call with converted constraints.
static CallInst *createNewInlineAsm(
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
reconstructReturnValue(Type *RetTy, CallInst *NewCall,
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

} // namespace

bool InlineAsmPrepare::runOnFunction(Function &F) {
  SmallVector<CallBase *, 4> IAs = findInlineAsms(F);
  if (IAs.empty())
    return false;

  bool Changed = false;
  for (CallBase *CB : IAs) {
    InlineAsm *IA = cast<InlineAsm>(CB->getCalledOperand());
    const InlineAsm::ConstraintInfoVector &Constraints = IA->ParseConstraints();

    auto [NewConstraintStr, HasRegMem] =
        convertConstraintsToMemory(IA->getConstraintString());
    if (!HasRegMem)
      continue;

    IRBuilder<> Builder(CB);
    IRBuilder<> EntryBuilder(&F.getEntryBlock(), F.getEntryBlock().begin());

    // Collect new arguments and return types.
    SmallVector<Value *, 8> NewArgs;
    SmallVector<Type *, 8> NewArgTypes;
    SmallVector<Type *, 2> NewRetTypes;
    SmallVector<std::pair<unsigned, Type *>, 8> ElementTypeAttrs;

    // Track allocas created for converted outputs.
    SmallVector<AllocaInst *, 8> OutputAllocas(Constraints.size(), nullptr);

    // Build tied constraint map.
    SmallVector<int, 8> TiedOutput(Constraints.size(), -1);
    buildTiedConstraintMap(Constraints, TiedOutput);

    // Process constraints.
    unsigned ArgNo = 0;
    unsigned OutputIdx = 0;
    for (unsigned I = 0, E = Constraints.size(); I != E; ++I) {
      const InlineAsm::ConstraintInfo &C = Constraints[I];

      if (C.Type == InlineAsm::isOutput) {
        processOutputConstraint(C, CB->getType(), OutputIdx, EntryBuilder,
                                NewArgs, NewArgTypes, NewRetTypes,
                                ElementTypeAttrs, OutputAllocas, I);
        OutputIdx++;
      } else if (C.Type == InlineAsm::isInput) {
        Value *ArgVal = CB->getArgOperand(ArgNo);
        processInputConstraint(C, ArgVal, TiedOutput, OutputAllocas, I, Builder,
                               EntryBuilder, NewArgs, NewArgTypes);
        ArgNo++;
      }
    }

    // Build the new return type.
    Type *NewRetTy = buildReturnType(NewRetTypes, F.getContext());

    // Create the new inline assembly call.
    CallInst *NewCall =
        createNewInlineAsm(IA, NewConstraintStr, NewRetTy, NewArgTypes, NewArgs,
                           ElementTypeAttrs, CB, Builder, F.getContext());

    // Reconstruct the return value and update users.
    if (!CB->use_empty()) {
      if (Value *Replacement =
              reconstructReturnValue(CB->getType(), NewCall, Constraints,
                                     OutputAllocas, NewRetTypes, Builder))
        CB->replaceAllUsesWith(Replacement);
    }

    CB->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses InlineAsmPreparePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  bool Changed = InlineAsmPrepare().runOnFunction(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
