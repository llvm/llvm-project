//===- BottomUpVec.cpp - A bottom-up vectorizer pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm::sandboxir {

BottomUpVec::BottomUpVec(StringRef Pipeline)
    : FunctionPass("bottom-up-vec"),
      RPM("rpm", Pipeline, SandboxVectorizerPassBuilder::createRegionPass) {}

// TODO: This is a temporary function that returns some seeds.
//       Replace this with SeedCollector's function when it lands.
static llvm::SmallVector<Value *, 4> collectSeeds(BasicBlock &BB) {
  llvm::SmallVector<Value *, 4> Seeds;
  for (auto &I : BB)
    if (auto *SI = llvm::dyn_cast<StoreInst>(&I))
      Seeds.push_back(SI);
  return Seeds;
}

static SmallVector<Value *, 4> getOperand(ArrayRef<Value *> Bndl,
                                          unsigned OpIdx) {
  SmallVector<Value *, 4> Operands;
  for (Value *BndlV : Bndl) {
    auto *BndlI = cast<Instruction>(BndlV);
    Operands.push_back(BndlI->getOperand(OpIdx));
  }
  return Operands;
}

static BasicBlock::iterator
getInsertPointAfterInstrs(ArrayRef<Value *> Instrs) {
  // TODO: Use the VecUtils function for getting the bottom instr once it lands.
  auto *BotI = cast<Instruction>(
      *std::max_element(Instrs.begin(), Instrs.end(), [](auto *V1, auto *V2) {
        return cast<Instruction>(V1)->comesBefore(cast<Instruction>(V2));
      }));
  // If Bndl contains Arguments or Constants, use the beginning of the BB.
  return std::next(BotI->getIterator());
}

Value *BottomUpVec::createVectorInstr(ArrayRef<Value *> Bndl,
                                      ArrayRef<Value *> Operands) {
  Change = true;
  assert(all_of(Bndl, [](auto *V) { return isa<Instruction>(V); }) &&
         "Expect Instructions!");
  auto &Ctx = Bndl[0]->getContext();

  Type *ScalarTy = VecUtils::getElementType(Utils::getExpectedType(Bndl[0]));
  auto *VecTy = VecUtils::getWideType(ScalarTy, VecUtils::getNumLanes(Bndl));

  BasicBlock::iterator WhereIt = getInsertPointAfterInstrs(Bndl);

  auto Opcode = cast<Instruction>(Bndl[0])->getOpcode();
  switch (Opcode) {
  case Instruction::Opcode::ZExt:
  case Instruction::Opcode::SExt:
  case Instruction::Opcode::FPToUI:
  case Instruction::Opcode::FPToSI:
  case Instruction::Opcode::FPExt:
  case Instruction::Opcode::PtrToInt:
  case Instruction::Opcode::IntToPtr:
  case Instruction::Opcode::SIToFP:
  case Instruction::Opcode::UIToFP:
  case Instruction::Opcode::Trunc:
  case Instruction::Opcode::FPTrunc:
  case Instruction::Opcode::BitCast: {
    assert(Operands.size() == 1u && "Casts are unary!");
    return CastInst::create(VecTy, Opcode, Operands[0], WhereIt, Ctx, "VCast");
  }
  case Instruction::Opcode::FCmp:
  case Instruction::Opcode::ICmp: {
    auto Pred = cast<CmpInst>(Bndl[0])->getPredicate();
    assert(all_of(drop_begin(Bndl),
                  [Pred](auto *SBV) {
                    return cast<CmpInst>(SBV)->getPredicate() == Pred;
                  }) &&
           "Expected same predicate across bundle.");
    return CmpInst::create(Pred, Operands[0], Operands[1], WhereIt, Ctx,
                           "VCmp");
  }
  case Instruction::Opcode::Select: {
    return SelectInst::create(Operands[0], Operands[1], Operands[2], WhereIt,
                              Ctx, "Vec");
  }
  case Instruction::Opcode::FNeg: {
    auto *UOp0 = cast<UnaryOperator>(Bndl[0]);
    auto OpC = UOp0->getOpcode();
    return UnaryOperator::createWithCopiedFlags(OpC, Operands[0], UOp0, WhereIt,
                                                Ctx, "Vec");
  }
  case Instruction::Opcode::Add:
  case Instruction::Opcode::FAdd:
  case Instruction::Opcode::Sub:
  case Instruction::Opcode::FSub:
  case Instruction::Opcode::Mul:
  case Instruction::Opcode::FMul:
  case Instruction::Opcode::UDiv:
  case Instruction::Opcode::SDiv:
  case Instruction::Opcode::FDiv:
  case Instruction::Opcode::URem:
  case Instruction::Opcode::SRem:
  case Instruction::Opcode::FRem:
  case Instruction::Opcode::Shl:
  case Instruction::Opcode::LShr:
  case Instruction::Opcode::AShr:
  case Instruction::Opcode::And:
  case Instruction::Opcode::Or:
  case Instruction::Opcode::Xor: {
    auto *BinOp0 = cast<BinaryOperator>(Bndl[0]);
    auto *LHS = Operands[0];
    auto *RHS = Operands[1];
    return BinaryOperator::createWithCopiedFlags(BinOp0->getOpcode(), LHS, RHS,
                                                 BinOp0, WhereIt, Ctx, "Vec");
  }
  case Instruction::Opcode::Load: {
    auto *Ld0 = cast<LoadInst>(Bndl[0]);
    Value *Ptr = Ld0->getPointerOperand();
    return LoadInst::create(VecTy, Ptr, Ld0->getAlign(), WhereIt, Ctx, "VecL");
  }
  case Instruction::Opcode::Store: {
    auto Align = cast<StoreInst>(Bndl[0])->getAlign();
    Value *Val = Operands[0];
    Value *Ptr = Operands[1];
    return StoreInst::create(Val, Ptr, Align, WhereIt, Ctx);
  }
  case Instruction::Opcode::Br:
  case Instruction::Opcode::Ret:
  case Instruction::Opcode::PHI:
  case Instruction::Opcode::AddrSpaceCast:
  case Instruction::Opcode::Call:
  case Instruction::Opcode::GetElementPtr:
    llvm_unreachable("Unimplemented");
    break;
  default:
    llvm_unreachable("Unimplemented");
    break;
  }
  llvm_unreachable("Missing switch case!");
  // TODO: Propagate debug info.
}

Value *BottomUpVec::vectorizeRec(ArrayRef<Value *> Bndl) {
  Value *NewVec = nullptr;
  const auto &LegalityRes = Legality->canVectorize(Bndl);
  switch (LegalityRes.getSubclassID()) {
  case LegalityResultID::Widen: {
    auto *I = cast<Instruction>(Bndl[0]);
    SmallVector<Value *, 2> VecOperands;
    switch (I->getOpcode()) {
    case Instruction::Opcode::Load:
      // Don't recurse towards the pointer operand.
      VecOperands.push_back(cast<LoadInst>(I)->getPointerOperand());
      break;
    case Instruction::Opcode::Store: {
      // Don't recurse towards the pointer operand.
      auto *VecOp = vectorizeRec(getOperand(Bndl, 0));
      VecOperands.push_back(VecOp);
      VecOperands.push_back(cast<StoreInst>(I)->getPointerOperand());
      break;
    }
    default:
      // Visit all operands.
      for (auto OpIdx : seq<unsigned>(I->getNumOperands())) {
        auto *VecOp = vectorizeRec(getOperand(Bndl, OpIdx));
        VecOperands.push_back(VecOp);
      }
      break;
    }
    NewVec = createVectorInstr(Bndl, VecOperands);

    // TODO: Collect potentially dead instructions.
    break;
  }
  case LegalityResultID::Pack: {
    // TODO: Unimplemented
    llvm_unreachable("Unimplemented");
  }
  }
  return NewVec;
}

bool BottomUpVec::tryVectorize(ArrayRef<Value *> Bndl) {
  vectorizeRec(Bndl);
  return Change;
}

bool BottomUpVec::runOnFunction(Function &F, const Analyses &A) {
  Legality = std::make_unique<LegalityAnalysis>(
      A.getAA(), A.getScalarEvolution(), F.getParent()->getDataLayout(),
      F.getContext());
  Change = false;
  // TODO: Start from innermost BBs first
  for (auto &BB : F) {
    // TODO: Replace with proper SeedCollector function.
    auto Seeds = collectSeeds(BB);
    // TODO: Slice Seeds into smaller chunks.
    // TODO: If vectorization succeeds, run the RegionPassManager on the
    // resulting region.
    if (Seeds.size() >= 2)
      Change |= tryVectorize(Seeds);
  }
  return Change;
}

} // namespace llvm::sandboxir
