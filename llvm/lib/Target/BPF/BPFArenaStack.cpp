//===-- BPFArenaStack.cpp - Keep locals in arena --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In a program that keeps its memory in arena a pointer to a local has to be
// like other pointers:
// - An insn is either an access to arena or to the stack. A function that takes
//   a pointer cannot be called with a local in one place and with what is
//   on the heap in another.
// - A pointer to the stack that is stored in arena is a number when it's loaded
//   back.
// - The kernel has to see that the access is within the stack. The bounds check
//   that the compiler proved to be redundant is not there for it to see.
// - What a function does to the stack of its callers is a part of what it does,
//   so the kernel verifies it again for every caller.
//
// So locals are in arena too. The kernel gives a stack in arena to every
// invocation of the program. It grows up, so that r12 is to the frame of
// the function there what r10 is to its frame in the stack of the program:
//
//   r12 += N                   the first insn of the function
//   *(u64 *)(r12 - 8) = r1     locals are at r12 - N ... r12 - 1
//   r2 = r12                   the address of a local is a number, like other
//   r2 += -16                  addresses in arena
//
// The kernel drops the frame when the function exits. N is a multiple of 16.
//
// What is left on the stack of the program:
// - Spilled registers. They are not in the IR.
// - What is passed to helpers and kfuncs: they take memory that the kernel
//   knows about.
// - Locals that hold pointers of the kernel. BPFArenaKptr.cpp marks them.
//
// This pass picks the locals. They become objects of another stack, see
// TargetStackID::BPFArena, when insns are selected.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-stack"

static cl::list<std::string> ArenaKfuncArgs(
    "bpf-arena-kfunc-args", cl::Hidden, cl::CommaSeparated,
    cl::desc("Arguments of kfuncs that point to arena: name:number, from 0. "
             "In the kernel the names of such parameters end in __arena"));

bool llvm::isBPFArenaArg(const CallBase &CB, const Use &U) {
  const Function *F = CB.getCalledFunction();

  if (!F || ArenaKfuncArgs.empty() || !CB.isArgOperand(&U))
    return false;
  return is_contained(
      ArenaKfuncArgs,
      (F->getName() + ":" + Twine(CB.getArgOperandNo(&U))).str());
}

/// A helper or a kfunc. memcmp() and the like are not: they end up as functions
/// of the program or as kfuncs that take pointers to arena.
bool llvm::isBPFKernelFunc(const CallBase &CB, const TargetLibraryInfo &TLI) {
  const Value *Callee = CB.getCalledOperand();

  if (CB.isInlineAsm() || isa<IntrinsicInst>(CB))
    return false;
  if (const auto *F = dyn_cast<Function>(Callee))
    return F->isDeclaration() && TLI.getLibFunc(*F) == NotLibFunc;
  // The number of a helper.
  return isa<Constant>(Callee);
}

/// Helpers and kfuncs take memory that the kernel knows about.
bool llvm::bpfStaysOnStack(const AllocaInst &AI, const TargetLibraryInfo &TLI) {
  SmallVector<const Value *, 8> Work = {&AI};

  while (!Work.empty()) {
    for (const Use &U : Work.pop_back_val()->uses()) {
      if (const auto *GEP = dyn_cast<GetElementPtrInst>(U.getUser())) {
        if (GEP->hasAllConstantIndices())
          Work.push_back(GEP);
      } else if (const auto *CB = dyn_cast<CallBase>(U.getUser())) {
        if (CB->isArgOperand(&U) && isBPFKernelFunc(*CB, TLI) &&
            !isBPFArenaArg(*CB, U))
          return true;
      }
    }
  }
  return false;
}

/// r12 is aligned to 16 bytes. Make room for a local that wants more and align
/// its address when the function runs.
static AllocaInst *realign(AllocaInst &AI) {
  constexpr uint64_t StackAlign = 16;
  const DataLayout &DL = AI.getDataLayout();
  uint64_t Align = AI.getAlign().value();
  std::optional<TypeSize> Size = AI.getAllocationSize(DL);

  if (Align <= StackAlign || !Size || Size->isScalable())
    return &AI;

  IRBuilder<> B(&AI);
  Type *I8 = B.getInt8Ty();
  AllocaInst *Raw = B.CreateAlloca(
      ArrayType::get(I8, Size->getFixedValue() + Align - StackAlign));
  Raw->setAlignment(llvm::Align(StackAlign));
  Raw->takeName(&AI);
  Value *P =
      B.CreateIntrinsic(Intrinsic::ptrmask, {B.getPtrTy(), B.getInt64Ty()},
                        {B.CreatePtrAdd(Raw, B.getInt64(Align - StackAlign)),
                         B.getInt64(-(int64_t)Align)});

  // They are for allocas only.
  for (User *U : make_early_inc_range(AI.users()))
    if (auto *II = dyn_cast<LifetimeIntrinsic>(U))
      II->eraseFromParent();
  AI.replaceAllUsesWith(P);
  AI.eraseFromParent();
  return Raw;
}

static bool markAllocas(Function &F, const TargetLibraryInfo &TLI) {
  MDNode *MD = MDNode::get(F.getContext(), {});
  SmallVector<AllocaInst *, 8> Allocas;

  for (Instruction &I : instructions(F)) {
    auto *AI = dyn_cast<AllocaInst>(&I);

    // Dynamic allocas are not supported on either stack.
    if (AI && AI->isStaticAlloca() && !AI->hasMetadata(BPFArenaStackMD) &&
        !AI->hasMetadata(BPFStackKeepMD) && !bpfStaysOnStack(*AI, TLI))
      Allocas.push_back(AI);
  }
  for (AllocaInst *AI : Allocas)
    realign(*AI)->setMetadata(BPFArenaStackMD, MD);
  return !Allocas.empty();
}

namespace {

struct BPFArenaStackLegacy : public FunctionPass {
  static char ID;
  BPFArenaStackLegacy() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Stack"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override {
    return markAllocas(F,
                       getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F));
  }
};

} // namespace

char BPFArenaStackLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(BPFArenaStackLegacy, DEBUG_TYPE, "BPF Arena Stack", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(BPFArenaStackLegacy, DEBUG_TYPE, "BPF Arena Stack", false,
                    false)

FunctionPass *llvm::createBPFArenaStackPass() {
  return new BPFArenaStackLegacy();
}

PreservedAnalyses BPFArenaStackPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  if (!markAllocas(F, AM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
