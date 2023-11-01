//===--- VectorWiden.cpp - Combining Vector Operations to wider types ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to widen vector operations to a wider type, it finds
// independent from each other operations with a certain vector type as SLP does
// with scalars by Bottom Up. It detects consecutive stores that can be put
// together into a wider vector-stores. Next, it attempts to construct
// vectorizable tree using the use-def chains.
//
//==------------------------------------------------------------------------==//

#include "llvm/Transforms/Vectorize/VectorWiden.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "vector-widen"

// Due to independant operations to widening that we consider with possibility
// to merge those operations into one and also to widening store if we find
// later store instructions. We have to consider the distance between those
// independent operations or we might introduce bad register pressure, etc.

static cl::opt<unsigned>
    MaxInstDistance("vw-max-instr-distance", cl::init(30), cl::Hidden,
                    cl::desc("Maximum distance between instructions to"
                             "consider to widen"));

static cl::opt<bool> OverrideTargetConsiderToWiden(
    "vw-override-target-consider-to-widen", cl::init(false), cl::Hidden,
    cl::desc("Ignore any target information while considoring to widen"));

namespace {
class VectorWiden {
public:
  using InstrList = SmallVector<Instruction *, 2>;
  using ValueList = SmallVector<Value *, 2>;
  VectorWiden(Function &F, const TargetTransformInfo &TTI)
      : F(F), Builder(F.getContext()), TTI(TTI) {}

  bool run();

private:
  Function &F;
  IRBuilder<> Builder;
  const TargetTransformInfo &TTI;
  TargetLibraryInfo *TLI;

  DenseSet<Instruction *> DeletedInstructions;

  /// Checks if the instruction is marked for deletion.
  bool isDeleted(Instruction *I) const { return DeletedInstructions.count(I); }

  /// Removes an instruction from its block and eventually deletes it.
  void eraseInstruction(Instruction *I) { DeletedInstructions.insert(I); }

  bool processBB(BasicBlock &BB, LLVMContext &Context);

  bool canWidenNode(ArrayRef<Instruction *> IL, LLVMContext &Context);

  bool widenNode(ArrayRef<Instruction *> IL, LLVMContext &Context);

  void widenCastInst(ArrayRef<Instruction *> IL);

  void widenBinaryOperator(ArrayRef<Instruction *> IL);

  InstructionCost getOpCost(unsigned Opcode, Type *To, Type *From,
                            Instruction *I);
};
} // namespace

void VectorWiden::widenCastInst(ArrayRef<Instruction *> IL) {
  Instruction *I = IL[0];
  Instruction *I1 = IL[1];
  auto *RetOrigType = cast<VectorType>(I->getType());
  auto *OrigType = cast<VectorType>(I->getOperand(0)->getType());
  auto *RetType = VectorType::getDoubleElementsVectorType(RetOrigType);
  auto *OpType = VectorType::getDoubleElementsVectorType(OrigType);

  bool isBitCast = I->getOpcode() == Instruction::BitCast;
  unsigned Offset =
      dyn_cast<ScalableVectorType>(OrigType)
          ? (cast<ScalableVectorType>(OrigType))->getMinNumElements()
          : (cast<FixedVectorType>(OrigType))->getNumElements();
  unsigned BitCastOffsetExtract =
      (dyn_cast<ScalableVectorType>(RetType)
           ? (cast<ScalableVectorType>(RetType))->getMinNumElements()
           : (cast<FixedVectorType>(RetType))->getNumElements()) /
      2;
  Value *WideVec = UndefValue::get(OpType);
  Builder.SetInsertPoint(I);
  Function *InsertIntr = llvm::Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::vector_insert, {OpType, OrigType});
  Value *Insert1 = Builder.CreateCall(
      InsertIntr, {WideVec, I->getOperand(0), Builder.getInt64(0)});
  Value *Insert2 = Builder.CreateCall(
      InsertIntr, {Insert1, I1->getOperand(0), Builder.getInt64(Offset)});
  Value *ResCast = Builder.CreateCast(Instruction::CastOps(I->getOpcode()),
                                      Insert2, RetType);

  Function *ExtractIntr = llvm::Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::vector_extract, {RetOrigType, RetType});
  if (!I->users().empty()) {
    Value *Res =
        Builder.CreateCall(ExtractIntr, {ResCast, Builder.getInt64(0)});
    I->replaceAllUsesWith(Res);
  }
  if (!I1->users().empty()) {
    Value *Res = Builder.CreateCall(
        ExtractIntr,
        {ResCast, Builder.getInt64(isBitCast ? BitCastOffsetExtract : Offset)});
    I1->replaceAllUsesWith(Res);
  }
}

void VectorWiden::widenBinaryOperator(ArrayRef<Instruction *> IL) {
  Instruction *I = IL[0];
  Instruction *I1 = IL[1];

  Value *XHi = I->getOperand(0);
  Value *XLo = I1->getOperand(0);
  Value *YHi = I->getOperand(1);
  Value *YLo = I1->getOperand(1);

  auto *RetOrigType = cast<VectorType>(I->getType());
  auto *OrigType = cast<VectorType>(I->getOperand(0)->getType());
  auto *RetType = VectorType::getDoubleElementsVectorType(RetOrigType);
  auto *OpType = VectorType::getDoubleElementsVectorType(OrigType);
  unsigned Offset =
      dyn_cast<ScalableVectorType>(OrigType)
          ? (cast<ScalableVectorType>(OrigType))->getMinNumElements()
          : (cast<FixedVectorType>(OrigType))->getNumElements();
  Value *WideVec = UndefValue::get(OpType);
  Builder.SetInsertPoint(I);
  Function *InsertIntr = llvm::Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::vector_insert, {OpType, OrigType});
  Value *X1 =
      Builder.CreateCall(InsertIntr, {WideVec, XLo, Builder.getInt64(0)});
  Value *X2 =
      Builder.CreateCall(InsertIntr, {X1, XHi, Builder.getInt64(Offset)});
  Value *Y1 =
      Builder.CreateCall(InsertIntr, {WideVec, YLo, Builder.getInt64(0)});
  Value *Y2 =
      Builder.CreateCall(InsertIntr, {Y1, YHi, Builder.getInt64(Offset)});
  Value *ResBinOp =
      Builder.CreateBinOp((Instruction::BinaryOps)I->getOpcode(), X2, Y2);
  ValueList VL;
  for (Instruction *I : IL)
    VL.push_back(I);

  propagateIRFlags(ResBinOp, VL);

  Function *ExtractIntr = llvm::Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::vector_extract, {RetOrigType, RetType});
  if (!I->users().empty()) {
    Value *Res =
        Builder.CreateCall(ExtractIntr, {ResBinOp, Builder.getInt64(Offset)});
    I->replaceAllUsesWith(Res);
  }
  if (!I1->users().empty()) {
    Value *Res =
        Builder.CreateCall(ExtractIntr, {ResBinOp, Builder.getInt64(0)});
    I1->replaceAllUsesWith(Res);
  }
}

bool VectorWiden::canWidenNode(ArrayRef<Instruction *> IL,
                               LLVMContext &Context) {
  if (!OverrideTargetConsiderToWiden && !TTI.considerToWiden(Context, IL))
    return false;

  bool HasSecondOperand = IL[0]->getNumOperands() > 1;
  for (int X = 0, E = IL.size(); X < E; X++) {
    for (int Y = 0, E = IL.size(); Y < E; Y++) {
      if (X == Y)
        continue;
      if (IL[X] == IL[Y] || IL[X]->getOperand(0) == IL[Y] ||
          (HasSecondOperand && IL[X]->getOperand(1) == IL[Y]))
        return false;
    }
    if (isDeleted(IL[X]) || !IL[X]->hasOneUse())
      return false;
    if (X == 0)
      continue;
    if (IL[X]->getOpcode() != IL[X - 1]->getOpcode() ||
        // Ignore if any types are different.
        IL[X]->getType() != IL[X - 1]->getType() ||
        IL[X]->getOperand(0)->getType() !=
            IL[X - 1]->getOperand(0)->getType() ||
        IL[X - 1]->comesBefore(IL[X]))
      return false;
    if (IL[0]->getParent() == IL[X]->user_back()->getParent() &&
        IL[X]->user_back()->comesBefore(IL[0]))
      return false;
  }
  return true;
}

bool VectorWiden::widenNode(ArrayRef<Instruction *> IL, LLVMContext &Context) {
  // Currently, this pass supports only two operations to widen to
  // a single operation.
  if (IL.size() != 2)
    return false;
  if (!canWidenNode(IL, Context))
    return false;

  unsigned Opcode = IL[0]->getOpcode();

  if (dyn_cast<CastInst>(IL[0])) {
    if (!OverrideTargetConsiderToWiden) {
      auto *OrigType = cast<VectorType>(IL[0]->getOperand(0)->getType());
      auto *RetOrigType = cast<VectorType>(IL[0]->getType());
      InstructionCost Cost = getOpCost(Opcode, RetOrigType, OrigType, IL[0]);
      auto *RetType = VectorType::getDoubleElementsVectorType(RetOrigType);
      auto *OpType = VectorType::getDoubleElementsVectorType(OrigType);
      InstructionCost CostNew = getOpCost(Opcode, RetType, OpType, IL[0]);
      if (2 * Cost < CostNew)
        return false;
    }
    LLVM_DEBUG(
        dbgs()
        << "VW: Decided to widen CastInst, safe to merge node starting with "
        << *IL[0] << "\n");
    widenCastInst(IL);
    return true;
  }
  if (dyn_cast<BinaryOperator>(IL[0])) {
    if (!OverrideTargetConsiderToWiden) {
      auto *OrigType = cast<VectorType>(IL[0]->getOperand(0)->getType());
      auto *OpType = VectorType::getDoubleElementsVectorType(OrigType);
      InstructionCost Cost = getOpCost(Opcode, OrigType, OrigType, IL[0]);
      InstructionCost CostNew = getOpCost(Opcode, OpType, OpType, IL[0]);
      if (2 * Cost < CostNew)
        return false;
    }
    LLVM_DEBUG(
        dbgs()
        << "VW: Decided to widen BinaryOp, safe to merge node starting with "
        << *IL[0] << "\n");
    // We want to propagate here IR flags for the group of operations like
    // "fast" flag for float pointer ones or "nuw" for integer instructions.
    widenBinaryOperator(IL);
    return true;
  }
  return false;
}

InstructionCost VectorWiden::getOpCost(unsigned Opcode, Type *To, Type *From,
                                       Instruction *I) {
  InstructionCost Cost = 0;
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  if (dyn_cast<BinaryOperator>(I)) {
    unsigned OpIdx = isa<UnaryOperator>(I) ? 0 : 1;
    TTI::OperandValueInfo Op1Info = TTI::getOperandInfo(I->getOperand(0));
    TTI::OperandValueInfo Op2Info = TTI::getOperandInfo(I->getOperand(OpIdx));
    SmallVector<const Value *> Operands(I->operand_values());
    Cost = TTI.getArithmeticInstrCost(I->getOpcode(), To, CostKind, Op1Info,
                                      Op2Info, Operands, I);
  } else if (dyn_cast<CastInst>(I)) {
    Cost = TTI.getCastInstrCost(Opcode, To, From, TTI::getCastContextHint(I),
                                CostKind, I);
  }
  return Cost;
}

static bool isOperationSupported(Instruction *I) {
  unsigned Opcode = I->getOpcode();
  // Currently, we support only those operations, but later we could add more.
  if (dyn_cast<VectorType>(I->getType()) &&
      (I->isBinaryOp() || Opcode == Instruction::SExt ||
       Opcode == Instruction::ZExt || Opcode == Instruction::FPToUI ||
       Opcode == Instruction::FPToSI || Opcode == Instruction::FPExt ||
       Opcode == Instruction::SIToFP || Opcode == Instruction::UIToFP ||
       Opcode == Instruction::Trunc || Opcode == Instruction::FPTrunc ||
       Opcode == Instruction::BitCast))
    return true;
  return false;
}

bool VectorWiden::processBB(BasicBlock &BB, LLVMContext &Context) {
  struct Operation {
    // Position where the first operation, in the list of operations,
    // was discovered and the last instruction in the current basic block.
    unsigned Position;
    InstrList Ops;
  };
  // The key is opertion opcode.
  // The value is a list of operations with the first operation position in
  // the basic block.
  DenseMap<unsigned, Operation> Operations;
  Instruction *LastInstr = BB.getTerminator();
  unsigned CurrentPosition = 0;
  for (BasicBlock::reverse_iterator IP(BB.rbegin()); IP != BB.rend();
       *IP++, ++CurrentPosition) {
    Instruction *I = &*IP;
    unsigned OpFound = 0;

    if (I->isDebugOrPseudoInst() || isDeleted(I) || !isOperationSupported(I))
      continue;

    unsigned Opcode = I->getOpcode();
    if (Operations.contains(Opcode)) {
      Operation *OpRec = &Operations[Opcode];
      // If instructions are too apart then remove old instruction
      // and reset position to the next instruction in the list instruction.
      if (CurrentPosition - OpRec->Position > MaxInstDistance) {
        unsigned NumToDelete = 0;
        for (InstrList::iterator It = OpRec->Ops.begin();
             It != OpRec->Ops.end(); ++It) {
          Instruction *Instr = *It;
          unsigned NewPosition =
              std::distance(Instr->getIterator(), LastInstr->getIterator());
          if (CurrentPosition - NewPosition > MaxInstDistance) {
            NumToDelete++;
          } else {
            // Updating Position value to next remaining in range opertion.
            OpRec->Position = NewPosition;
            LLVM_DEBUG(dbgs() << "VW: Updating node starting with "
                              << **(OpRec->Ops.begin())
                              << " position to : " << NewPosition << "\n");
            break;
          }
        }
        for (unsigned i = 0; i < NumToDelete; ++i) {
          LLVM_DEBUG(dbgs()
                     << "VW: Deleting operation " << **(OpRec->Ops.begin())
                     << " from node as out of range."
                     << "\n");
          OpRec->Ops.erase(OpRec->Ops.begin());
        }
      }
      // If no operations left in the list, set position to the current.
      if (!OpRec->Ops.size())
        OpRec->Position = CurrentPosition;
      OpRec->Ops.push_back(I);
      LLVM_DEBUG(dbgs() << "VW: Found operation " << *I
                        << " to add to existing node starting at "
                        << **(OpRec->Ops.begin()) << " at : " << OpRec->Position
                        << "\n");
      if (OpRec->Ops.size() > 1)
        OpFound = Opcode;
    } else {
      LLVM_DEBUG(dbgs() << "VW: Found operation " << *I
                        << " to form a node at : " << CurrentPosition << "\n");
      Operations[Opcode] = {CurrentPosition, {I}};
    }

    if (OpFound && Operations.contains(OpFound)) {
      auto *OpRec = &Operations[OpFound];
      for (Instruction *Op : OpRec->Ops)
        LLVM_DEBUG(dbgs() << "VW: operation to check : " << *Op << "\n");
      if (!widenNode(OpRec->Ops, Context)) {
        LLVM_DEBUG(dbgs() << "VW: Unable use a wider vector for vector ops.\n");
        if (OpRec->Ops.size() > 4) {
          LLVM_DEBUG(dbgs() << "VW: Deleting operation "
                            << **(OpRec->Ops.begin()) << " as unable to widen."
                            << "\n");
          OpRec->Ops.erase(OpRec->Ops.begin());
          OpRec->Position = std::distance(
              (*(OpRec->Ops.begin()))->getIterator(), LastInstr->getIterator());
        }
      } else {
        for (Instruction *Instr : OpRec->Ops)
          eraseInstruction(Instr);
        return true;
      }
    }
  }
  return false;
}

bool VectorWiden::run() {
  bool Changed = false;
  LLVMContext &Context = F.getContext();

  LLVM_DEBUG(dbgs() << "VW: Function:" << F.getName() << "\n");
  for (BasicBlock &BB : F) {
    LLVM_DEBUG(dbgs() << "VW: BB:" << BB.getName() << "\n");

    // If any transformation is done, then we have to start all over again,
    // since we generate new instructions.
    while (processBB(BB, Context))
      Changed = true;
  }

  if (Changed)
    for (auto *I : DeletedInstructions)
      RecursivelyDeleteTriviallyDeadInstructions(I);

  return Changed;
}

PreservedAnalyses VectorWidenPass::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);

  VectorWiden VecWiden(F, TTI);

  if (!VecWiden.run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
