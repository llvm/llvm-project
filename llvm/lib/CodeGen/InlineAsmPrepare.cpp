//===-- InlineAsmPrepare - Prepare inline asm for code generation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InlineAsmPrepare.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "inline-asm-prepare"

namespace {

class InlineAsmPrepare : public FunctionPass {
  InlineAsmPrepare(InlineAsmPrepare &) = delete;

public:
  InlineAsmPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
  bool runOnFunction(Function &F) override;

  static char ID;
};

char InlineAsmPrepare::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(InlineAsmPrepare, DEBUG_TYPE,
                "Convert inline asm \"rm\" insts for fast register allocation",
                false, false)
FunctionPass *llvm::createInlineAsmPass() { return new InlineAsmPrepare(); }

// For each inline asm, the "rm" constraint needs to default to "m" for the
// fast register allocator.
static SmallVector<CallBase *, 4> findInlineAsms(Function &F) {
  SmallVector<CallBase *, 4> InlineAsms;

  for (auto &BB : F) {
    for (auto &I : BB) {
      CallBase *CB = dyn_cast<CallBase>(&I);
      if (!CB || !CB->isInlineAsm())
        continue;
      InlineAsms.push_back(CB);
    }
  }

  return InlineAsms;
}

static bool isRegMemConstraint(StringRef Constraint) {
  return Constraint.size() == 2 && (Constraint == "rm" || Constraint == "mr");
}

// Convert instances of the "rm" constraints into "m".
static std::string convertConstraintsToMemory(StringRef ConstraintStr) {
  auto I = ConstraintStr.begin(), E = ConstraintStr.end();
  std::ostringstream Out;

  while (I != E) {
    bool IsOutput = false;
    bool HasIndirect = false;
    if (*I == '=') {
      Out << *I;
      IsOutput = true;
      ++I;
    }
    if (*I == '*') {
      Out << '*';
      HasIndirect = true;
      ++I;
    }
    if (*I == '+') {
      Out << '+';
      IsOutput = true;
      ++I;
    }

    auto Comma = std::find(I, E, ',');
    std::string Sub(I, Comma);
    if (isRegMemConstraint(Sub)) {
      if (IsOutput && !HasIndirect)
        Out << '*';
      Out << 'm';
    } else {
      Out << Sub;
    }

    if (Comma == E)
      break;

    Out << ',';
    I = Comma + 1;
  }

  return Out.str();
}

bool InlineAsmPrepare::runOnFunction(Function &F) {
  // Only process "rm" on x86 platforms.
  if (!F.getParent()->getTargetTriple().isX86())
    return false;

  SmallVector<CallBase *, 4> IAs = findInlineAsms(F);
  if (IAs.empty())
    return false;

  bool Changed = false;
  for (CallBase *CB : IAs) {
    InlineAsm *IA = cast<InlineAsm>(CB->getCalledOperand());
    const InlineAsm::ConstraintInfoVector &Constraints = IA->ParseConstraints();

    std::string NewConstraintStr =
        convertConstraintsToMemory(IA->getConstraintString());
    if (NewConstraintStr == IA->getConstraintString())
      continue;

    IRBuilder<> Builder(CB);
    // IRBuilder<> EntryBuilder(&F.getEntryBlock(), F.getEntryBlock().begin());

    // Collect new arguments and return types.
    SmallVector<Value *, 8> NewArgs;
    SmallVector<Type *, 8> NewArgTypes;
    SmallVector<Type *, 2> NewRetTypes;

    SmallVector<std::pair<unsigned, Type *>, 8> ElementTypeAttrs;

    // Track allocas created for converted outputs.
    // Maps constraint index to the AllocaInst created for it (if any).
    SmallVector<AllocaInst *, 8> OutputAllocas(Constraints.size(), nullptr);

    // Track pairs of Input-Output tied constraints.
    // TiedOutput[i] = j means Constraint i is an Input tied to Output
    // Constraint j.
    SmallVector<int, 8> TiedOutput(Constraints.size(), -1);
    for (unsigned I = 0, E = Constraints.size(); I != E; ++I) {
      const auto &C = Constraints[I];
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

    unsigned ArgNo = 0;
    unsigned OutputIdx = 0;
    for (unsigned I = 0, E = Constraints.size(); I != E; ++I) {
      const auto &C = Constraints[I];

      if (C.Type == InlineAsm::isOutput) {
        // Output-only or Output with matching input (Read-Write)
        Type *RetTy = CB->getType();
        Type *SlotTy = RetTy;

        if (StructType *ST = dyn_cast<StructType>(RetTy))
          SlotTy = ST->getElementType(OutputIdx);

        if (C.hasRegMemConstraints()) {
          // Converted to memory constraint. Create alloca and pass pointer as
          // argument.
          AllocaInst *Slot = Builder.CreateAlloca(SlotTy, nullptr, "asm_mem");
          NewArgs.push_back(Slot);
          NewArgTypes.push_back(Slot->getType());
          ElementTypeAttrs.push_back({NewArgs.size() - 1, SlotTy});
          OutputAllocas[I] = Slot;
          // No return value for this output since it's now an out-parameter.
        } else {
          // Unchanged, still an output return value.
          NewRetTypes.push_back(SlotTy);
        }

        OutputIdx++;
      } else if (C.Type == InlineAsm::isInput) {
        // Input
        Value *ArgVal = CB->getArgOperand(ArgNo);
        Type *ArgTy = ArgVal->getType();
        bool Handled = false;

        if (TiedOutput[I] != -1) {
          int MatchIdx = TiedOutput[I];
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
            AllocaInst *Slot = Builder.CreateAlloca(ArgTy, nullptr, "asm_mem");
            Builder.CreateStore(ArgVal, Slot);
            NewArgs.push_back(Slot);
            NewArgTypes.push_back(Slot->getType());
          } else {
            // Unchanged
            NewArgs.push_back(ArgVal);
            NewArgTypes.push_back(ArgTy);
          }
        }
        ArgNo++;
      }
    }

    Type *NewRetTy = nullptr;
    if (NewRetTypes.empty()) {
      NewRetTy = Type::getVoidTy(F.getContext());
    } else if (NewRetTypes.size() == 1) {
      NewRetTy = NewRetTypes[0];
    } else {
      NewRetTy = StructType::get(F.getContext(), NewRetTypes);
    }

    FunctionType *NewFTy = FunctionType::get(NewRetTy, NewArgTypes, false);
    auto *NewIA = InlineAsm::get(NewFTy, IA->getAsmString(), NewConstraintStr,
                                 IA->hasSideEffects(), IA->isAlignStack(),
                                 IA->getDialect(), IA->canThrow());

    CallInst *NewCall = Builder.CreateCall(NewFTy, NewIA, NewArgs);
    NewCall->setCallingConv(CB->getCallingConv());
    NewCall->setAttributes(CB->getAttributes());
    NewCall->setDebugLoc(CB->getDebugLoc());

    for (const auto &Item : ElementTypeAttrs)
      NewCall->addParamAttr(
          Item.first,
          Attribute::get(F.getContext(), Attribute::ElementType, Item.second));

    // Reconstruct the return value and update users.
    if (!CB->use_empty()) {
      Value *Replacement = nullptr;
      Type *RetTy = CB->getType();

      if (RetTy->isVoidTy()) {
        // No return value, nothing to replace.
      } else if (isa<StructType>(RetTy)) {
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
        Replacement = Res;
      } else {
        // Single output.
        // Find the output constraint (should be the first one).
        unsigned OutConstraintIdx = 0;
        for (unsigned I = 0; I < Constraints.size(); ++I) {
          if (Constraints[I].Type == InlineAsm::isOutput) {
            OutConstraintIdx = I;
            break;
          }
        }

        if (AllocaInst *Slot = OutputAllocas[OutConstraintIdx]) {
          Replacement = Builder.CreateLoad(Slot->getAllocatedType(), Slot);
        } else {
          Replacement = NewCall;
        }
      }

      if (Replacement) {
        CB->replaceAllUsesWith(Replacement);
      }
    }

    CB->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses InlineAsmPreparePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  InlineAsmPrepare IAP;

  bool Changed = IAP.runOnFunction(F);
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::all();
}
