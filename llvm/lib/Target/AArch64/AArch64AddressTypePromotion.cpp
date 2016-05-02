//===-- AArch64AddressTypePromotion.cpp --- Promote type for addr accesses -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass tries to promote the computations use to obtained a sign extended
// value used into memory accesses.
// E.g.
// a = add nsw i32 b, 3
// d = sext i32 a to i64
// e = getelementptr ..., i64 d
//
// =>
// f = sext i32 b to i64
// a = add nsw i64 f, 3
// e = getelementptr ..., i64 a
//
// This is legal to do if the computations are marked with either nsw or nuw
// markers. Moreover, the current heuristic is simple: it does not create new
// sext operations, i.e., it gives up when a sext would have forked (e.g., if a
// = add i32 b, c, two sexts are required to promote the computation).
//
// FIXME: This pass may be useful for other targets too.
// ===---------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-type-promotion"

static cl::opt<bool>
EnableAddressTypePromotion("aarch64-type-promotion", cl::Hidden,
                           cl::desc("Enable the type promotion pass"),
                           cl::init(true));
static cl::opt<bool>
EnableMerge("aarch64-type-promotion-merge", cl::Hidden,
            cl::desc("Enable merging of redundant sexts when one is dominating"
                     " the other."),
            cl::init(true));

#define AARCH64_TYPE_PROMO_NAME "AArch64 Address Type Promotion"

//===----------------------------------------------------------------------===//
//                       AArch64AddressTypePromotion
//===----------------------------------------------------------------------===//

namespace llvm {
void initializeAArch64AddressTypePromotionPass(PassRegistry &);
}

namespace {
class AArch64AddressTypePromotion : public FunctionPass {

public:
  static char ID;
  AArch64AddressTypePromotion()
      : FunctionPass(ID), Func(nullptr), ConsideredSExtType(nullptr) {
    initializeAArch64AddressTypePromotionPass(*PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override {
    return AARCH64_TYPE_PROMO_NAME;
  }

  /// Iterate over the functions and promote the computation of interesting
  // sext instructions.
  bool runOnFunction(Function &F) override;

private:
  /// The current function.
  Function *Func;
  /// Filter out all sexts that does not have this type.
  /// Currently initialized with Int64Ty.
  Type *ConsideredSExtType;

  // This transformation requires dominator info.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  typedef SmallPtrSet<Instruction *, 32> SetOfInstructions;
  typedef SmallVector<Instruction *, 16> Instructions;
  typedef DenseMap<Value *, Instructions> ValueToInsts;

  /// Check if it is profitable to move a sext through this instruction.
  /// Currently, we consider it is profitable if:
  /// - Inst is used only once (no need to insert truncate).
  /// - Inst has only one operand that will require a sext operation (we do
  ///   do not create new sext operation).
  bool shouldGetThrough(const Instruction *Inst);

  /// Check if it is possible and legal to move a sext through this
  /// instruction.
  /// Current heuristic considers that we can get through:
  /// - Arithmetic operation marked with the nsw or nuw flag.
  /// - Other sext operation.
  /// - Truncate operation if it was just dropping sign extended bits.
  bool canGetThrough(const Instruction *Inst);

  /// Move sext operations through safe to sext instructions.
  bool propagateSignExtension(Instructions &SExtInsts);

  /// Is this sext should be considered for code motion.
  /// We look for sext with ConsideredSExtType and uses in at least one
  // GetElementPtrInst.
  bool shouldConsiderSExt(const Instruction *SExt) const;

  /// Collect all interesting sext operations, i.e., the ones with the right
  /// type and used in memory accesses.
  /// More precisely, a sext instruction is considered as interesting if it
  /// is used in a "complex" getelementptr or it exits at least another
  /// sext instruction that sign extended the same initial value.
  /// A getelementptr is considered as "complex" if it has more than 2
  // operands.
  void analyzeSExtension(Instructions &SExtInsts);

  /// Merge redundant sign extension operations in common dominator.
  void mergeSExts(ValueToInsts &ValToSExtendedUses,
                  SetOfInstructions &ToRemove);
};
} // end anonymous namespace.

char AArch64AddressTypePromotion::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64AddressTypePromotion, "aarch64-type-promotion",
                      AARCH64_TYPE_PROMO_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AArch64AddressTypePromotion, "aarch64-type-promotion",
                    AARCH64_TYPE_PROMO_NAME, false, false)

FunctionPass *llvm::createAArch64AddressTypePromotionPass() {
  return new AArch64AddressTypePromotion();
}

bool AArch64AddressTypePromotion::canGetThrough(const Instruction *Inst) {
  if (isa<SExtInst>(Inst))
    return true;

  const BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Inst);
  if (BinOp && isa<OverflowingBinaryOperator>(BinOp) &&
      (BinOp->hasNoUnsignedWrap() || BinOp->hasNoSignedWrap()))
    return true;

  // sext(trunc(sext)) --> sext
  if (isa<TruncInst>(Inst) && isa<SExtInst>(Inst->getOperand(0))) {
    const Instruction *Opnd = cast<Instruction>(Inst->getOperand(0));
    // Check that the truncate just drop sign extended bits.
    if (Inst->getType()->getIntegerBitWidth() >=
            Opnd->getOperand(0)->getType()->getIntegerBitWidth() &&
        Inst->getOperand(0)->getType()->getIntegerBitWidth() <=
            ConsideredSExtType->getIntegerBitWidth())
      return true;
  }

  return false;
}

bool AArch64AddressTypePromotion::shouldGetThrough(const Instruction *Inst) {
  // If the type of the sext is the same as the considered one, this sext
  // will become useless.
  // Otherwise, we will have to do something to preserve the original value,
  // unless it is used once.
  if (isa<SExtInst>(Inst) &&
      (Inst->getType() == ConsideredSExtType || Inst->hasOneUse()))
    return true;

  // If the Inst is used more that once, we may need to insert truncate
  // operations and we don't do that at the moment.
  if (!Inst->hasOneUse())
    return false;

  // This truncate is used only once, thus if we can get thourgh, it will become
  // useless.
  if (isa<TruncInst>(Inst))
    return true;

  // If both operands are not constant, a new sext will be created here.
  // Current heuristic is: each step should be profitable.
  // Therefore we don't allow to increase the number of sext even if it may
  // be profitable later on.
  if (isa<BinaryOperator>(Inst) && isa<ConstantInt>(Inst->getOperand(1)))
    return true;

  return false;
}

static bool shouldSExtOperand(const Instruction *Inst, int OpIdx) {
  return !(isa<SelectInst>(Inst) && OpIdx == 0);
}

bool
AArch64AddressTypePromotion::shouldConsiderSExt(const Instruction *SExt) const {
  if (SExt->getType() != ConsideredSExtType)
    return false;

  for (const User *U : SExt->users()) {
    if (isa<GetElementPtrInst>(U))
      return true;
  }

  return false;
}

// Input:
// - SExtInsts contains all the sext instructions that are used directly in
//   GetElementPtrInst, i.e., access to memory.
// Algorithm:
// - For each sext operation in SExtInsts:
//   Let var be the operand of sext.
//   while it is profitable (see shouldGetThrough), legal, and safe
//   (see canGetThrough) to move sext through var's definition:
//   * promote the type of var's definition.
//   * fold var into sext uses.
//   * move sext above var's definition.
//   * update sext operand to use the operand of var that should be sign
//     extended (by construction there is only one).
//
//   E.g.,
//   a = ... i32 c, 3
//   b = sext i32 a to i64 <- is it legal/safe/profitable to get through 'a'
//   ...
//   = b
// => Yes, update the code
//   b = sext i32 c to i64
//   a = ... i64 b, 3
//   ...
//   = a
// Iterate on 'c'.
bool
AArch64AddressTypePromotion::propagateSignExtension(Instructions &SExtInsts) {
  DEBUG(dbgs() << "*** Propagate Sign Extension ***\n");

  bool LocalChange = false;
  SetOfInstructions ToRemove;
  ValueToInsts ValToSExtendedUses;
  while (!SExtInsts.empty()) {
    // Get through simple chain.
    Instruction *SExt = SExtInsts.pop_back_val();

    DEBUG(dbgs() << "Consider:\n" << *SExt << '\n');

    // If this SExt has already been merged continue.
    if (SExt->use_empty() && ToRemove.count(SExt)) {
      DEBUG(dbgs() << "No uses => marked as delete\n");
      continue;
    }

    // Now try to get through the chain of definitions.
    while (auto *Inst = dyn_cast<Instruction>(SExt->getOperand(0))) {
      DEBUG(dbgs() << "Try to get through:\n" << *Inst << '\n');
      if (!canGetThrough(Inst) || !shouldGetThrough(Inst)) {
        // We cannot get through something that is not an Instruction
        // or not safe to SExt.
        DEBUG(dbgs() << "Cannot get through\n");
        break;
      }

      LocalChange = true;
      // If this is a sign extend, it becomes useless.
      if (isa<SExtInst>(Inst) || isa<TruncInst>(Inst)) {
        DEBUG(dbgs() << "SExt or trunc, mark it as to remove\n");
        // We cannot use replaceAllUsesWith here because we may trigger some
        // assertion on the type as all involved sext operation may have not
        // been moved yet.
        while (!Inst->use_empty()) {
          Use &U = *Inst->use_begin();
          Instruction *User = dyn_cast<Instruction>(U.getUser());
          assert(User && "User of sext is not an Instruction!");
          User->setOperand(U.getOperandNo(), SExt);
        }
        ToRemove.insert(Inst);
        SExt->setOperand(0, Inst->getOperand(0));
        SExt->moveBefore(Inst);
        continue;
      }

      // Get through the Instruction:
      // 1. Update its type.
      // 2. Replace the uses of SExt by Inst.
      // 3. Sign extend each operand that needs to be sign extended.

      // Step #1.
      Inst->mutateType(SExt->getType());
      // Step #2.
      SExt->replaceAllUsesWith(Inst);
      // Step #3.
      Instruction *SExtForOpnd = SExt;

      DEBUG(dbgs() << "Propagate SExt to operands\n");
      for (int OpIdx = 0, EndOpIdx = Inst->getNumOperands(); OpIdx != EndOpIdx;
           ++OpIdx) {
        DEBUG(dbgs() << "Operand:\n" << *(Inst->getOperand(OpIdx)) << '\n');
        if (Inst->getOperand(OpIdx)->getType() == SExt->getType() ||
            !shouldSExtOperand(Inst, OpIdx)) {
          DEBUG(dbgs() << "No need to propagate\n");
          continue;
        }
        // Check if we can statically sign extend the operand.
        Value *Opnd = Inst->getOperand(OpIdx);
        if (const ConstantInt *Cst = dyn_cast<ConstantInt>(Opnd)) {
          DEBUG(dbgs() << "Statically sign extend\n");
          Inst->setOperand(OpIdx, ConstantInt::getSigned(SExt->getType(),
                                                         Cst->getSExtValue()));
          continue;
        }
        // UndefValue are typed, so we have to statically sign extend them.
        if (isa<UndefValue>(Opnd)) {
          DEBUG(dbgs() << "Statically sign extend\n");
          Inst->setOperand(OpIdx, UndefValue::get(SExt->getType()));
          continue;
        }

        // Otherwise we have to explicity sign extend it.
        assert(SExtForOpnd &&
               "Only one operand should have been sign extended");

        SExtForOpnd->setOperand(0, Opnd);

        DEBUG(dbgs() << "Move before:\n" << *Inst << "\nSign extend\n");
        // Move the sign extension before the insertion point.
        SExtForOpnd->moveBefore(Inst);
        Inst->setOperand(OpIdx, SExtForOpnd);
        // If more sext are required, new instructions will have to be created.
        SExtForOpnd = nullptr;
      }
      if (SExtForOpnd == SExt) {
        DEBUG(dbgs() << "Sign extension is useless now\n");
        ToRemove.insert(SExt);
        break;
      }
    }

    // If the use is already of the right type, connect its uses to its argument
    // and delete it.
    // This can happen for an Instruction all uses of which are sign extended.
    if (!ToRemove.count(SExt) &&
        SExt->getType() == SExt->getOperand(0)->getType()) {
      DEBUG(dbgs() << "Sign extension is useless, attach its use to "
                      "its argument\n");
      SExt->replaceAllUsesWith(SExt->getOperand(0));
      ToRemove.insert(SExt);
    } else
      ValToSExtendedUses[SExt->getOperand(0)].push_back(SExt);
  }

  if (EnableMerge)
    mergeSExts(ValToSExtendedUses, ToRemove);

  // Remove all instructions marked as ToRemove.
  for (Instruction *I: ToRemove)
    I->eraseFromParent();
  return LocalChange;
}

void AArch64AddressTypePromotion::mergeSExts(ValueToInsts &ValToSExtendedUses,
                                             SetOfInstructions &ToRemove) {
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  for (auto &Entry : ValToSExtendedUses) {
    Instructions &Insts = Entry.second;
    Instructions CurPts;
    for (Instruction *Inst : Insts) {
      if (ToRemove.count(Inst))
        continue;
      bool inserted = false;
      for (auto &Pt : CurPts) {
        if (DT.dominates(Inst, Pt)) {
          DEBUG(dbgs() << "Replace all uses of:\n" << *Pt << "\nwith:\n"
                       << *Inst << '\n');
          Pt->replaceAllUsesWith(Inst);
          ToRemove.insert(Pt);
          Pt = Inst;
          inserted = true;
          break;
        }
        if (!DT.dominates(Pt, Inst))
          // Give up if we need to merge in a common dominator as the
          // expermients show it is not profitable.
          continue;

        DEBUG(dbgs() << "Replace all uses of:\n" << *Inst << "\nwith:\n"
                     << *Pt << '\n');
        Inst->replaceAllUsesWith(Pt);
        ToRemove.insert(Inst);
        inserted = true;
        break;
      }
      if (!inserted)
        CurPts.push_back(Inst);
    }
  }
}

void AArch64AddressTypePromotion::analyzeSExtension(Instructions &SExtInsts) {
  DEBUG(dbgs() << "*** Analyze Sign Extensions ***\n");

  DenseMap<Value *, Instruction *> SeenChains;

  for (auto &BB : *Func) {
    for (auto &II : BB) {
      Instruction *SExt = &II;

      // Collect all sext operation per type.
      if (!isa<SExtInst>(SExt) || !shouldConsiderSExt(SExt))
        continue;

      DEBUG(dbgs() << "Found:\n" << (*SExt) << '\n');

      // Cases where we actually perform the optimization:
      // 1. SExt is used in a getelementptr with more than 2 operand =>
      //    likely we can merge some computation if they are done on 64 bits.
      // 2. The beginning of the SExt chain is SExt several time. =>
      //    code sharing is possible.

      bool insert = false;
      // #1.
      for (const User *U : SExt->users()) {
        const Instruction *Inst = dyn_cast<GetElementPtrInst>(U);
        if (Inst && Inst->getNumOperands() > 2) {
          DEBUG(dbgs() << "Interesting use in GetElementPtrInst\n" << *Inst
                       << '\n');
          insert = true;
          break;
        }
      }

      // #2.
      // Check the head of the chain.
      Instruction *Inst = SExt;
      Value *Last;
      do {
        int OpdIdx = 0;
        const BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Inst);
        if (BinOp && isa<ConstantInt>(BinOp->getOperand(0)))
          OpdIdx = 1;
        Last = Inst->getOperand(OpdIdx);
        Inst = dyn_cast<Instruction>(Last);
      } while (Inst && canGetThrough(Inst) && shouldGetThrough(Inst));

      DEBUG(dbgs() << "Head of the chain:\n" << *Last << '\n');
      DenseMap<Value *, Instruction *>::iterator AlreadySeen =
          SeenChains.find(Last);
      if (insert || AlreadySeen != SeenChains.end()) {
        DEBUG(dbgs() << "Insert\n");
        SExtInsts.push_back(SExt);
        if (AlreadySeen != SeenChains.end() && AlreadySeen->second != nullptr) {
          DEBUG(dbgs() << "Insert chain member\n");
          SExtInsts.push_back(AlreadySeen->second);
          SeenChains[Last] = nullptr;
        }
      } else {
        DEBUG(dbgs() << "Record its chain membership\n");
        SeenChains[Last] = SExt;
      }
    }
  }
}

bool AArch64AddressTypePromotion::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  if (!EnableAddressTypePromotion || F.isDeclaration())
    return false;
  Func = &F;
  ConsideredSExtType = Type::getInt64Ty(Func->getContext());

  DEBUG(dbgs() << "*** " << getPassName() << ": " << Func->getName() << '\n');

  Instructions SExtInsts;
  analyzeSExtension(SExtInsts);
  return propagateSignExtension(SExtInsts);
}
