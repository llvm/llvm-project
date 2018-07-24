//===----- ARMCodeGenPrepare.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass inserts intrinsics to handle small types that would otherwise be
/// promoted during legalization. Here we can manually promote types or insert
/// intrinsics which can handle narrow types that aren't supported by the
/// register classes.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "arm-codegenprepare"

using namespace llvm;

static cl::opt<bool>
DisableCGP("arm-disable-cgp", cl::Hidden, cl::init(true),
           cl::desc("Disable ARM specific CodeGenPrepare pass"));

static cl::opt<bool>
EnableDSP("arm-enable-scalar-dsp", cl::Hidden, cl::init(false),
          cl::desc("Use DSP instructions for scalar operations"));

static cl::opt<bool>
EnableDSPWithImms("arm-enable-scalar-dsp-imms", cl::Hidden, cl::init(false),
                   cl::desc("Use DSP instructions for scalar operations\
                            with immediate operands"));

namespace {

class IRPromoter {
  SmallPtrSet<Value*, 8> NewInsts;
  SmallVector<Instruction*, 4> InstsToRemove;
  Module *M = nullptr;
  LLVMContext &Ctx;

public:
  IRPromoter(Module *M) : M(M), Ctx(M->getContext()) { }

  void Cleanup() {
    for (auto *I : InstsToRemove) {
      LLVM_DEBUG(dbgs() << "ARM CGP: Removing " << *I << "\n");
      I->dropAllReferences();
      I->eraseFromParent();
    }
    InstsToRemove.clear();
    NewInsts.clear();
  }

  void Mutate(Type *OrigTy,
              SmallPtrSetImpl<Value*> &Visited,
              SmallPtrSetImpl<Value*> &Leaves,
              SmallPtrSetImpl<Instruction*> &Roots);
};

class ARMCodeGenPrepare : public FunctionPass {
  const ARMSubtarget *ST = nullptr;
  IRPromoter *Promoter = nullptr;
  std::set<Value*> AllVisited;
  Type *OrigTy = nullptr;
  unsigned TypeSize = 0;

  bool isNarrowInstSupported(Instruction *I);
  bool isSupportedValue(Value *V);
  bool isLegalToPromote(Value *V);
  bool TryToPromote(Value *V);

public:
  static char ID;

  ARMCodeGenPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }

  StringRef getPassName() const override { return "ARM IR optimizations"; }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  bool doFinalization(Module &M) override;
};

}

/// Can the given value generate sign bits.
static bool isSigned(Value *V) {
  if (!isa<Instruction>(V))
    return false;

  unsigned Opc = cast<Instruction>(V)->getOpcode();
  return Opc == Instruction::AShr || Opc == Instruction::SDiv ||
         Opc == Instruction::SRem;
}

/// Some instructions can use 8- and 16-bit operands, and we don't need to
/// promote anything larger. We disallow booleans to make life easier when
/// dealing with icmps but allow any other integer that is <= 16 bits. Void
/// types are accepted so we can handle switches.
static bool isSupportedType(Value *V) {
  if (V->getType()->isVoidTy())
    return true;

  const IntegerType *IntTy = dyn_cast<IntegerType>(V->getType());
  if (!IntTy)
    return false;

  // Don't try to promote boolean values.
  if (IntTy->getBitWidth() == 1)
    return false;

  if (auto *ZExt = dyn_cast<ZExtInst>(V))
    return isSupportedType(ZExt->getOperand(0));

  return IntTy->getBitWidth() <= 16;
}

/// Return true if V will require any promoted values to be truncated for the
/// use to be valid.
static bool isSink(Value *V) {
  auto UsesNarrowValue = [](Value *V) {
    return V->getType()->getScalarSizeInBits() <= 32;
  };

  if (auto *Store = dyn_cast<StoreInst>(V))
    return UsesNarrowValue(Store->getValueOperand());
  if (auto *Return = dyn_cast<ReturnInst>(V))
    return UsesNarrowValue(Return->getReturnValue());

  return isa<CallInst>(V);
}

/// Return true if the given value is a leaf that will need to be zext'd.
static bool isSource(Value *V) {
  if (isa<Argument>(V) && isSupportedType(V))
    return true;
  else if (isa<TruncInst>(V))
    return true;
  else if (auto *ZExt = dyn_cast<ZExtInst>(V))
    // ZExt can be a leaf if its the only user of a load.
    return isa<LoadInst>(ZExt->getOperand(0)) &&
                         ZExt->getOperand(0)->hasOneUse();
  else if (auto *Call = dyn_cast<CallInst>(V))
    return Call->hasRetAttr(Attribute::AttrKind::ZExt);
  else if (auto *Load = dyn_cast<LoadInst>(V)) {
    if (!isa<IntegerType>(Load->getType()))
      return false;
    // A load is a leaf, unless its already just being zext'd.
    if (Load->hasOneUse() && isa<ZExtInst>(*Load->use_begin()))
      return false;

    return true;
  }
  return false;
}

/// Return whether the instruction can be promoted within any modifications to
/// it's operands or result.
static bool isSafeOverflow(Instruction *I) {
  if (isa<OverflowingBinaryOperator>(I) && I->hasNoUnsignedWrap())
    return true;

  unsigned Opc = I->getOpcode();
  if (Opc == Instruction::Add || Opc == Instruction::Sub) {
    // We don't care if the add or sub could wrap if the value is decreasing
    // and is only being used by an unsigned compare.
    if (!I->hasOneUse() ||
        !isa<ICmpInst>(*I->user_begin()) ||
        !isa<ConstantInt>(I->getOperand(1)))
      return false;

    auto *CI = cast<ICmpInst>(*I->user_begin());
    if (CI->isSigned())
      return false;

    bool NegImm = cast<ConstantInt>(I->getOperand(1))->isNegative();
    bool IsDecreasing = ((Opc == Instruction::Sub) && !NegImm) ||
                        ((Opc == Instruction::Add) && NegImm);
    if (!IsDecreasing)
      return false;

    LLVM_DEBUG(dbgs() << "ARM CGP: Allowing safe overflow for " << *I << "\n");
    return true;
  }

  // Otherwise, if an instruction is using a negative immediate we will need
  // to fix it up during the promotion.
  for (auto &Op : I->operands()) {
    if (auto *Const = dyn_cast<ConstantInt>(Op))
      if (Const->isNegative())
        return false;
  }
  return false;
}

static bool shouldPromote(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  if (!isa<IntegerType>(V->getType()))
    return false;

  if (isa<StoreInst>(I) || isa<TerminatorInst>(I) || isa<TruncInst>(I) ||
      isa<ICmpInst>(I))
    return false;

  if (auto *ZExt = dyn_cast<ZExtInst>(I))
    return !ZExt->getDestTy()->isIntegerTy(32);

  return true;
}

/// Return whether we can safely mutate V's type to ExtTy without having to be
/// concerned with zero extending or truncation.
static bool isPromotedResultSafe(Value *V) {
  if (!isa<Instruction>(V))
    return true;

  if (isSigned(V))
    return false;

  // If I is only being used by something that will require its value to be
  // truncated, then we don't care about the promoted result.
  auto *I = cast<Instruction>(V);
  if (I->hasOneUse() && isSink(*I->use_begin()))
    return true;

  if (isa<OverflowingBinaryOperator>(I))
    return isSafeOverflow(I);
  return true;
}

/// Return the intrinsic for the instruction that can perform the same
/// operation but on a narrow type. This is using the parallel dsp intrinsics
/// on scalar values.
static Intrinsic::ID getNarrowIntrinsic(Instruction *I, unsigned TypeSize) {
  // Whether we use the signed or unsigned versions of these intrinsics
  // doesn't matter because we're not using the GE bits that they set in
  // the APSR.
  switch(I->getOpcode()) {
  default:
    break;
  case Instruction::Add:
    return TypeSize == 16 ? Intrinsic::arm_uadd16 :
      Intrinsic::arm_uadd8;
  case Instruction::Sub:
    return TypeSize == 16 ? Intrinsic::arm_usub16 :
      Intrinsic::arm_usub8;
  }
  llvm_unreachable("unhandled opcode for narrow intrinsic");
}

void IRPromoter::Mutate(Type *OrigTy,
                        SmallPtrSetImpl<Value*> &Visited,
                        SmallPtrSetImpl<Value*> &Leaves,
                        SmallPtrSetImpl<Instruction*> &Roots) {
  IRBuilder<> Builder{Ctx};
  Type *ExtTy = Type::getInt32Ty(M->getContext());
  unsigned TypeSize = OrigTy->getPrimitiveSizeInBits();
  SmallPtrSet<Value*, 8> Promoted;
  LLVM_DEBUG(dbgs() << "ARM CGP: Promoting use-def chains to from " << TypeSize
        << " to 32-bits\n");

  auto ReplaceAllUsersOfWith = [&](Value *From, Value *To) {
    SmallVector<Instruction*, 4> Users;
    Instruction *InstTo = dyn_cast<Instruction>(To);
    for (Use &U : From->uses()) {
      auto *User = cast<Instruction>(U.getUser());
      if (InstTo && User->isIdenticalTo(InstTo))
        continue;
      Users.push_back(User);
    }

    for (auto &U : Users)
      U->replaceUsesOfWith(From, To);
  };

  auto FixConst = [&](ConstantInt *Const, Instruction *I) {
    Constant *NewConst = nullptr;
    if (isSafeOverflow(I)) {
      NewConst = (Const->isNegative()) ?
        ConstantExpr::getSExt(Const, ExtTy) :
        ConstantExpr::getZExt(Const, ExtTy);
    } else {
      uint64_t NewVal = *Const->getValue().getRawData();
      if (Const->getType() == Type::getInt16Ty(Ctx))
        NewVal &= 0xFFFF;
      else
        NewVal &= 0xFF;
      NewConst = ConstantInt::get(ExtTy, NewVal);
    }
    I->replaceUsesOfWith(Const, NewConst);
  };

  auto InsertDSPIntrinsic = [&](Instruction *I) {
    LLVM_DEBUG(dbgs() << "ARM CGP: Inserting DSP intrinsic for "
               << *I << "\n");
    Function *DSPInst =
      Intrinsic::getDeclaration(M, getNarrowIntrinsic(I, TypeSize));
    Builder.SetInsertPoint(I);
    Builder.SetCurrentDebugLocation(I->getDebugLoc());
    Value *Args[] = { I->getOperand(0), I->getOperand(1) };
    CallInst *Call = Builder.CreateCall(DSPInst, Args);
    ReplaceAllUsersOfWith(I, Call);
    InstsToRemove.push_back(I);
    NewInsts.insert(Call);
  };

  auto InsertZExt = [&](Value *V, Instruction *InsertPt) {
    LLVM_DEBUG(dbgs() << "ARM CGP: Inserting ZExt for " << *V << "\n");
    Builder.SetInsertPoint(InsertPt);
    if (auto *I = dyn_cast<Instruction>(V))
      Builder.SetCurrentDebugLocation(I->getDebugLoc());
    auto *ZExt = cast<Instruction>(Builder.CreateZExt(V, ExtTy));
    if (isa<Argument>(V))
      ZExt->moveBefore(InsertPt);
    else
      ZExt->moveAfter(InsertPt);
    ReplaceAllUsersOfWith(V, ZExt);
    NewInsts.insert(ZExt);
  };

  // First, insert extending instructions between the leaves and their users.
  LLVM_DEBUG(dbgs() << "ARM CGP: Promoting leaves:\n");
  for (auto V : Leaves) {
    LLVM_DEBUG(dbgs() << " - " << *V << "\n");
    if (auto *ZExt = dyn_cast<ZExtInst>(V))
      ZExt->mutateType(ExtTy);
    else if (auto *I = dyn_cast<Instruction>(V))
      InsertZExt(I, I);
    else if (auto *Arg = dyn_cast<Argument>(V)) {
      BasicBlock &BB = Arg->getParent()->front();
      InsertZExt(Arg, &*BB.getFirstInsertionPt());
    } else {
      llvm_unreachable("unhandled leaf that needs extending");
    }
    Promoted.insert(V);
  }

  LLVM_DEBUG(dbgs() << "ARM CGP: Mutating the tree..\n");
  // Then mutate the types of the instructions within the tree. Here we handle
  // constant operands.
  for (auto *V : Visited) {
    if (Leaves.count(V))
      continue;

    if (!isa<Instruction>(V))
      continue;

    auto *I = cast<Instruction>(V);
    if (Roots.count(I))
      continue;

    for (auto &U : I->operands()) {
      if ((U->getType() == ExtTy) || !isSupportedType(&*U))
        continue;

      if (auto *Const = dyn_cast<ConstantInt>(&*U))
        FixConst(Const, I);
      else if (isa<UndefValue>(&*U))
        U->mutateType(ExtTy);
    }

    if (shouldPromote(I)) {
      I->mutateType(ExtTy);
      Promoted.insert(I);
    }
  }

  // Now we need to remove any zexts that have become unnecessary, as well
  // as insert any intrinsics.
  for (auto *V : Visited) {
    if (Leaves.count(V))
      continue;
    if (auto *ZExt = dyn_cast<ZExtInst>(V)) {
      if (ZExt->getDestTy() != ExtTy) {
        ZExt->mutateType(ExtTy);
        Promoted.insert(ZExt);
      }
      else if (ZExt->getSrcTy() == ExtTy) {
        ReplaceAllUsersOfWith(V, ZExt->getOperand(0));
        InstsToRemove.push_back(ZExt);
      }
      continue;
    }

    if (!shouldPromote(V) || isPromotedResultSafe(V))
      continue;

    // Replace unsafe instructions with appropriate intrinsic calls.
    InsertDSPIntrinsic(cast<Instruction>(V));
  }

  LLVM_DEBUG(dbgs() << "ARM CGP: Fixing up the roots:\n");
  // Fix up any stores or returns that use the results of the promoted
  // chain.
  for (auto I : Roots) {
    LLVM_DEBUG(dbgs() << " - " << *I << "\n");
    Type *TruncTy = OrigTy;
    if (auto *Store = dyn_cast<StoreInst>(I)) {
      auto *PtrTy = cast<PointerType>(Store->getPointerOperandType());
      TruncTy = PtrTy->getElementType();
    } else if (isa<ReturnInst>(I)) {
      Function *F = I->getParent()->getParent();
      TruncTy = F->getFunctionType()->getReturnType();
    }

    for (unsigned i = 0; i < I->getNumOperands(); ++i) {
      Value *V = I->getOperand(i);
      if (Promoted.count(V) || NewInsts.count(V)) {
        if (auto *Op = dyn_cast<Instruction>(V)) {

          if (auto *Call = dyn_cast<CallInst>(I))
            TruncTy = Call->getFunctionType()->getParamType(i);

          if (TruncTy == ExtTy)
            continue;

          LLVM_DEBUG(dbgs() << "ARM CGP: Creating " << *TruncTy
                     << " Trunc for " << *Op << "\n");
          Builder.SetInsertPoint(Op);
          auto *Trunc = cast<Instruction>(Builder.CreateTrunc(Op, TruncTy));
          Trunc->moveBefore(I);
          I->setOperand(i, Trunc);
          NewInsts.insert(Trunc);
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "ARM CGP: Mutation complete.\n");
}

bool ARMCodeGenPrepare::isNarrowInstSupported(Instruction *I) {
  if (!ST->hasDSP() || !EnableDSP || !isSupportedType(I))
    return false;

  if (ST->isThumb() && !ST->hasThumb2())
    return false;

  if (I->getOpcode() != Instruction::Add && I->getOpcode() != Instruction::Sub)
    return false;

  // TODO
  // Would it be profitable? For Thumb code, these parallel DSP instructions
  // are only Thumb-2, so we wouldn't be able to dual issue on Cortex-M33. For
  // Cortex-A, specifically Cortex-A72, the latency is double and throughput is
  // halved. They also do not take immediates as operands.
  for (auto &Op : I->operands()) {
    if (isa<Constant>(Op)) {
      if (!EnableDSPWithImms)
        return false;
    }
  }
  return true;
}

/// We accept most instructions, as well as Arguments and ConstantInsts. We
/// Disallow casts other than zext and truncs and only allow calls if their
/// return value is zeroext. We don't allow opcodes that can introduce sign
/// bits.
bool ARMCodeGenPrepare::isSupportedValue(Value *V) {
  LLVM_DEBUG(dbgs() << "ARM CGP: Is " << *V << " supported?\n");

  // Non-instruction values that we can handle.
  if (isa<ConstantInt>(V) || isa<Argument>(V))
    return true;

  // Memory instructions
  if (isa<StoreInst>(V) || isa<LoadInst>(V) || isa<GetElementPtrInst>(V))
    return true;

  // Branches and targets.
  if (auto *ICmp = dyn_cast<ICmpInst>(V))
    return ICmp->isEquality() || !ICmp->isSigned();

  if( isa<BranchInst>(V) || isa<SwitchInst>(V) || isa<BasicBlock>(V))
    return true;

  if (isa<PHINode>(V) || isa<SelectInst>(V) || isa<ReturnInst>(V))
    return true;

  // Special cases for calls as we need to check for zeroext
  // TODO We should accept calls even if they don't have zeroext, as they can
  // still be roots.
  if (auto *Call = dyn_cast<CallInst>(V))
    return Call->hasRetAttr(Attribute::AttrKind::ZExt);
  else if (auto *Cast = dyn_cast<CastInst>(V)) {
    if (isa<ZExtInst>(Cast))
      return Cast->getDestTy()->getScalarSizeInBits() <= 32;
    else if (auto *Trunc = dyn_cast<TruncInst>(V))
      return Trunc->getDestTy()->getScalarSizeInBits() <= TypeSize;
    else {
      LLVM_DEBUG(dbgs() << "ARM CGP: No, unsupported cast.\n");
      return false;
    }
  } else if (!isa<BinaryOperator>(V)) {
    LLVM_DEBUG(dbgs() << "ARM CGP: No, not a binary operator.\n");
    return false;
  }

  bool res = !isSigned(V);
  if (!res)
    LLVM_DEBUG(dbgs() << "ARM CGP: No, it's a signed instruction.\n");
  return res;
}

/// Check that the type of V would be promoted and that the original type is
/// smaller than the targeted promoted type. Check that we're not trying to
/// promote something larger than our base 'TypeSize' type.
bool ARMCodeGenPrepare::isLegalToPromote(Value *V) {
  if (!isSupportedType(V))
    return false;

  unsigned VSize = 0;
  if (auto *Ld = dyn_cast<LoadInst>(V)) {
    auto *PtrTy = cast<PointerType>(Ld->getPointerOperandType());
    VSize = PtrTy->getElementType()->getPrimitiveSizeInBits();
  } else if (auto *ZExt = dyn_cast<ZExtInst>(V)) {
    VSize = ZExt->getOperand(0)->getType()->getPrimitiveSizeInBits();
  } else {
    VSize = V->getType()->getPrimitiveSizeInBits();
  }

  if (VSize > TypeSize)
    return false;

  if (isPromotedResultSafe(V))
    return true;

  if (auto *I = dyn_cast<Instruction>(V))
    return isNarrowInstSupported(I);

  return false;
}

bool ARMCodeGenPrepare::TryToPromote(Value *V) {
  OrigTy = V->getType();
  TypeSize = OrigTy->getPrimitiveSizeInBits();

  if (!isSupportedValue(V) || !shouldPromote(V) || !isLegalToPromote(V))
    return false;

  LLVM_DEBUG(dbgs() << "ARM CGP: TryToPromote: " << *V << "\n");

  SetVector<Value*> WorkList;
  SmallPtrSet<Value*, 8> Leaves;
  SmallPtrSet<Instruction*, 4> Roots;
  WorkList.insert(V);
  SmallPtrSet<Value*, 16> CurrentVisited;
  CurrentVisited.clear();

  // Return true if the given value can, or has been, visited. Add V to the
  // worklist if needed.
  auto AddLegalInst = [&](Value *V) {
    if (CurrentVisited.count(V))
      return true;

    if (!isSupportedValue(V) || (shouldPromote(V) && !isLegalToPromote(V))) {
      LLVM_DEBUG(dbgs() << "ARM CGP: Can't handle: " << *V << "\n");
      return false;
    }

    WorkList.insert(V);
    return true;
  };

  // Iterate through, and add to, a tree of operands and users in the use-def.
  while (!WorkList.empty()) {
    Value *V = WorkList.back();
    WorkList.pop_back();
    if (CurrentVisited.count(V))
      continue;

    if (!isa<Instruction>(V) && !isSource(V))
      continue;

    // If we've already visited this value from somewhere, bail now because
    // the tree has already been explored.
    // TODO: This could limit the transform, ie if we try to promote something
    // from an i8 and fail first, before trying an i16.
    if (AllVisited.count(V)) {
      LLVM_DEBUG(dbgs() << "ARM CGP: Already visited this: " << *V << "\n");
      return false;
    }

    CurrentVisited.insert(V);
    AllVisited.insert(V);

    // Calls can be both sources and sinks.
    if (isSink(V))
      Roots.insert(cast<Instruction>(V));
    if (isSource(V))
      Leaves.insert(V);
    else if (auto *I = dyn_cast<Instruction>(V)) {
      // Visit operands of any instruction visited.
      for (auto &U : I->operands()) {
        if (!AddLegalInst(U))
          return false;
      }
    }

    // Don't visit users of a node which isn't going to be mutated unless its a
    // source.
    if (isSource(V) || shouldPromote(V)) {
      for (Use &U : V->uses()) {
        if (!AddLegalInst(U.getUser()))
          return false;
      }
    }
  }

  unsigned NumToPromote = 0;
  unsigned Cost = 0;
  for (auto *V : CurrentVisited) {
    // Truncs will cause a uxt and no zeroext arguments will often require
    // a uxt somewhere.
    if (isa<TruncInst>(V))
      ++Cost;
    else if (auto *Arg = dyn_cast<Argument>(V)) {
      if (!Arg->hasZExtAttr())
        ++Cost;
    }

    // Mem ops can automatically be extended/truncated and non-instructions
    // don't need anything done.
    if (Leaves.count(V) || isa<StoreInst>(V) || !isa<Instruction>(V))
      continue;

    // Will need to truncate calls args and returns.
    if (Roots.count(cast<Instruction>(V))) {
      ++Cost;
      continue;
    }

    if (shouldPromote(V))
      ++NumToPromote;
  }

  LLVM_DEBUG(dbgs() << "ARM CGP: Visited nodes:\n";
             for (auto *I : CurrentVisited)
               I->dump();
             );
  LLVM_DEBUG(dbgs() << "ARM CGP: Cost of promoting " << NumToPromote
             << " instructions = " << Cost << "\n");
  if (Cost > NumToPromote || (NumToPromote == 0))
    return false;

  Promoter->Mutate(OrigTy, CurrentVisited, Leaves, Roots);
  return true;
}

bool ARMCodeGenPrepare::doInitialization(Module &M) {
  Promoter = new IRPromoter(&M);
  return false;
}

bool ARMCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F) || DisableCGP)
    return false;

  auto *TPC = &getAnalysis<TargetPassConfig>();
  if (!TPC)
    return false;

  const TargetMachine &TM = TPC->getTM<TargetMachine>();
  ST = &TM.getSubtarget<ARMSubtarget>(F);
  bool MadeChange = false;
  LLVM_DEBUG(dbgs() << "ARM CGP: Running on " << F.getName() << "\n");

  // Search up from icmps to try to promote their operands.
  for (BasicBlock &BB : F) {
    auto &Insts = BB.getInstList();
    for (auto &I : Insts) {
      if (AllVisited.count(&I))
        continue;

      if (isa<ICmpInst>(I)) {
        auto &CI = cast<ICmpInst>(I);

        // Skip signed or pointer compares
        if (CI.isSigned() || !isa<IntegerType>(CI.getOperand(0)->getType()))
          continue;

        LLVM_DEBUG(dbgs() << "ARM CGP: Searching from: " << CI << "\n");
        for (auto &Op : CI.operands()) {
          if (auto *I = dyn_cast<Instruction>(Op)) {
            if (isa<ZExtInst>(I))
              MadeChange |= TryToPromote(I->getOperand(0));
            else
              MadeChange |= TryToPromote(I);
          }
        }
      }
    }
    Promoter->Cleanup();
    LLVM_DEBUG(if (verifyFunction(F, &dbgs())) {
                dbgs();
                report_fatal_error("Broken function after type promotion");
               });
  }
  if (MadeChange)
    LLVM_DEBUG(dbgs() << "After ARMCodeGenPrepare: " << F << "\n");

  return MadeChange;
}

bool ARMCodeGenPrepare::doFinalization(Module &M) {
  delete Promoter;
  return false;
}

INITIALIZE_PASS_BEGIN(ARMCodeGenPrepare, DEBUG_TYPE,
                      "ARM IR optimizations", false, false)
INITIALIZE_PASS_END(ARMCodeGenPrepare, DEBUG_TYPE, "ARM IR optimizations",
                    false, false)

char ARMCodeGenPrepare::ID = 0;

FunctionPass *llvm::createARMCodeGenPreparePass() {
  return new ARMCodeGenPrepare();
}
