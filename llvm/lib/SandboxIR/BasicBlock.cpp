//===- BasicBlock.cpp - The BasicBlock class of Sandbox IR ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Instruction.h"

namespace llvm::sandboxir {

BBIterator &BBIterator::operator++() {
  auto ItE = BB->end();
  assert(It != ItE && "Already at end!");
  ++It;
  if (It == ItE)
    return *this;
  Instruction &NextI = *cast<sandboxir::Instruction>(Ctx->getValue(&*It));
  unsigned Num = NextI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  It = std::next(It, Num - 1);
  return *this;
}

BBIterator &BBIterator::operator--() {
  assert(It != BB->begin() && "Already at begin!");
  if (It == BB->end()) {
    --It;
    return *this;
  }
  Instruction &CurrI = **this;
  unsigned Num = CurrI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  assert(std::prev(It, Num - 1) != BB->begin() && "Already at begin!");
  It = std::prev(It, Num);
  return *this;
}

BasicBlock *BBIterator::getNodeParent() const {
  llvm::BasicBlock *Parent = const_cast<BBIterator *>(this)->It.getNodeParent();
  return cast<BasicBlock>(Ctx->getValue(Parent));
}

BasicBlock::iterator::pointer
BasicBlock::iterator::getInstr(llvm::BasicBlock::iterator It) const {
  return cast_or_null<Instruction>(Ctx->getValue(&*It));
}

Function *BasicBlock::getParent() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  auto *F = BB->getParent();
  if (F == nullptr)
    // Detached
    return nullptr;
  return cast_or_null<Function>(Ctx.getValue(F));
}

void BasicBlock::buildBasicBlockFromLLVMIR(llvm::BasicBlock *LLVMBB) {
  for (llvm::Instruction &IRef : reverse(*LLVMBB)) {
    llvm::Instruction *I = &IRef;
    Ctx.getOrCreateValue(I);
    for (auto [OpIdx, Op] : enumerate(I->operands())) {
      // Skip instruction's label operands
      if (isa<llvm::BasicBlock>(Op))
        continue;
      // Skip metadata
      if (isa<llvm::MetadataAsValue>(Op))
        continue;
      // Skip asm
      if (isa<llvm::InlineAsm>(Op))
        continue;
      Ctx.getOrCreateValue(Op);
    }
  }
#if !defined(NDEBUG)
  verify();
#endif
}

BasicBlock::iterator BasicBlock::begin() const {
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(Val);
  llvm::BasicBlock::iterator It = BB->begin();
  if (!BB->empty()) {
    auto *V = Ctx.getValue(&*BB->begin());
    assert(V != nullptr && "No SandboxIR for BB->begin()!");
    auto *I = cast<Instruction>(V);
    unsigned Num = I->getNumOfIRInstrs();
    assert(Num >= 1u && "Bad getNumOfIRInstrs()");
    It = std::next(It, Num - 1);
  }
  return iterator(BB, It, &Ctx);
}

Instruction *BasicBlock::getTerminator() const {
  auto *TerminatorV =
      Ctx.getValue(cast<llvm::BasicBlock>(Val)->getTerminator());
  return cast_or_null<Instruction>(TerminatorV);
}

Instruction &BasicBlock::front() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<Instruction>(getContext().getValue(&*BB->begin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

Instruction &BasicBlock::back() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<Instruction>(getContext().getValue(&*BB->rbegin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

#ifndef NDEBUG
void BasicBlock::dumpOS(raw_ostream &OS) const {
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(Val);
  const auto &Name = BB->getName();
  OS << Name;
  if (!Name.empty())
    OS << ":\n";
  // If there are Instructions in the BB that are not mapped to SandboxIR, then
  // use a crash-proof dump.
  if (any_of(*BB, [this](llvm::Instruction &I) {
        return Ctx.getValue(&I) == nullptr;
      })) {
    OS << "<Crash-proof mode!>\n";
    DenseSet<Instruction *> Visited;
    for (llvm::Instruction &IRef : *BB) {
      Value *SBV = Ctx.getValue(&IRef);
      if (SBV == nullptr)
        OS << IRef << " *** No SandboxIR ***\n";
      else {
        auto *SBI = dyn_cast<Instruction>(SBV);
        if (SBI == nullptr) {
          OS << IRef << " *** Not a SBInstruction!!! ***\n";
        } else {
          if (Visited.insert(SBI).second)
            OS << *SBI << "\n";
        }
      }
    }
  } else {
    for (auto &SBI : *this) {
      SBI.dumpOS(OS);
      OS << "\n";
    }
  }
}

void BasicBlock::verify() const {
  assert(isa<llvm::BasicBlock>(Val) && "Expected BasicBlock!");
  for (const auto &I : *this) {
    I.verify();
  }
}
#endif // NDEBUG

} // namespace llvm::sandboxir
