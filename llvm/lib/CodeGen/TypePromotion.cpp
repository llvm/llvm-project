//===----- TypePromotion.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This is an opcode based type promotion pass for small types that would
/// otherwise be promoted during legalisation. This works around the limitations
/// of selection dag for cyclic regions. The search begins from icmp
/// instructions operands where a tree, consisting of non-wrapping or safe
/// wrapping instructions, is built, checked and promoted if possible.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "type-promotion"
#define PASS_NAME "Type Promotion"

using namespace llvm;

static cl::opt<bool>
DisablePromotion("disable-type-promotion", cl::Hidden, cl::init(false),
                 cl::desc("Disable type promotion pass"));

// The goal of this pass is to enable more efficient code generation for
// operations on narrow types (i.e. types with < 32-bits) and this is a
// motivating IR code example:
//
//   define hidden i32 @cmp(i8 zeroext) {
//     %2 = add i8 %0, -49
//     %3 = icmp ult i8 %2, 3
//     ..
//   }
//
// The issue here is that i8 is type-legalized to i32 because i8 is not a
// legal type. Thus, arithmetic is done in integer-precision, but then the
// byte value is masked out as follows:
//
//   t19: i32 = add t4, Constant:i32<-49>
//     t24: i32 = and t19, Constant:i32<255>
//
// Consequently, we generate code like this:
//
//   subs  r0, #49
//   uxtb  r1, r0
//   cmp r1, #3
//
// This shows that masking out the byte value results in generation of
// the UXTB instruction. This is not optimal as r0 already contains the byte
// value we need, and so instead we can just generate:
//
//   sub.w r1, r0, #49
//   cmp r1, #3
//
// We achieve this by type promoting the IR to i32 like so for this example:
//
//   define i32 @cmp(i8 zeroext %c) {
//     %0 = zext i8 %c to i32
//     %c.off = add i32 %0, -49
//     %1 = icmp ult i32 %c.off, 3
//     ..
//   }
//
// For this to be valid and legal, we need to prove that the i32 add is
// producing the same value as the i8 addition, and that e.g. no overflow
// happens.
//
// A brief sketch of the algorithm and some terminology.
// We pattern match interesting IR patterns:
// - which have "sources": instructions producing narrow values (i8, i16), and
// - they have "sinks": instructions consuming these narrow values.
//
// We collect all instruction connecting sources and sinks in a worklist, so
// that we can mutate these instruction and perform type promotion when it is
// legal to do so.

namespace {
class IRPromoter {
  LLVMContext &Ctx;
  IntegerType *OrigTy = nullptr;
  unsigned PromotedWidth = 0;
  SetVector<Value*> &Visited;
  SetVector<Value*> &Sources;
  SetVector<Instruction*> &Sinks;
  SmallVectorImpl<Instruction*> &SafeWrap;
  IntegerType *ExtTy = nullptr;
  SmallPtrSet<Value*, 8> NewInsts;
  SmallPtrSet<Instruction*, 4> InstsToRemove;
  DenseMap<Value*, SmallVector<Type*, 4>> TruncTysMap;
  SmallPtrSet<Value*, 8> Promoted;

  void ReplaceAllUsersOfWith(Value *From, Value *To);
  void PrepareWrappingAdds(void);
  void ExtendSources(void);
  void ConvertTruncs(void);
  void PromoteTree(void);
  void TruncateSinks(void);
  void Cleanup(void);

public:
  IRPromoter(LLVMContext &C, IntegerType *Ty, unsigned Width,
             SetVector<Value*> &visited, SetVector<Value*> &sources,
             SetVector<Instruction*> &sinks,
             SmallVectorImpl<Instruction*> &wrap) :
    Ctx(C), OrigTy(Ty), PromotedWidth(Width), Visited(visited),
    Sources(sources), Sinks(sinks), SafeWrap(wrap) {
    ExtTy = IntegerType::get(Ctx, PromotedWidth);
    assert(OrigTy->getPrimitiveSizeInBits() < ExtTy->getPrimitiveSizeInBits()
           && "Original type not smaller than extended type");
  }

  void Mutate();
};

class TypePromotion : public FunctionPass {
  unsigned TypeSize = 0;
  LLVMContext *Ctx = nullptr;
  unsigned RegisterBitWidth = 0;
  SmallPtrSet<Value*, 16> AllVisited;
  SmallPtrSet<Instruction*, 8> SafeToPromote;
  SmallVector<Instruction*, 4> SafeWrap;

  // Does V have the same size result type as TypeSize.
  bool EqualTypeSize(Value *V);
  // Does V have the same size, or narrower, result type as TypeSize.
  bool LessOrEqualTypeSize(Value *V);
  // Does V have a result type that is wider than TypeSize.
  bool GreaterThanTypeSize(Value *V);
  // Does V have a result type that is narrower than TypeSize.
  bool LessThanTypeSize(Value *V);
  // Should V be a leaf in the promote tree?
  bool isSource(Value *V);
  // Should V be a root in the promotion tree?
  bool isSink(Value *V);
  // Should we change the result type of V? It will result in the users of V
  // being visited.
  bool shouldPromote(Value *V);
  // Is I an add or a sub, which isn't marked as nuw, but where a wrapping
  // result won't affect the computation?
  bool isSafeWrap(Instruction *I);
  // Can V have its integer type promoted, or can the type be ignored.
  bool isSupportedType(Value *V);
  // Is V an instruction with a supported opcode or another value that we can
  // handle, such as constants and basic blocks.
  bool isSupportedValue(Value *V);
  // Is V an instruction thats result can trivially promoted, or has safe
  // wrapping.
  bool isLegalToPromote(Value *V);
  bool TryToPromote(Value *V, unsigned PromotedWidth);

public:
  static char ID;

  TypePromotion() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnFunction(Function &F) override;
};

}

static bool GenerateSignBits(Value *V) {
  if (!isa<Instruction>(V))
    return false;

  unsigned Opc = cast<Instruction>(V)->getOpcode();
  return Opc == Instruction::AShr || Opc == Instruction::SDiv ||
         Opc == Instruction::SRem || Opc == Instruction::SExt;
}

bool TypePromotion::EqualTypeSize(Value *V) {
  return V->getType()->getScalarSizeInBits() == TypeSize;
}

bool TypePromotion::LessOrEqualTypeSize(Value *V) {
  return V->getType()->getScalarSizeInBits() <= TypeSize;
}

bool TypePromotion::GreaterThanTypeSize(Value *V) {
  return V->getType()->getScalarSizeInBits() > TypeSize;
}

bool TypePromotion::LessThanTypeSize(Value *V) {
  return V->getType()->getScalarSizeInBits() < TypeSize;
}

/// Return true if the given value is a source in the use-def chain, producing
/// a narrow 'TypeSize' value. These values will be zext to start the promotion
/// of the tree to i32. We guarantee that these won't populate the upper bits
/// of the register. ZExt on the loads will be free, and the same for call
/// return values because we only accept ones that guarantee a zeroext ret val.
/// Many arguments will have the zeroext attribute too, so those would be free
/// too.
bool TypePromotion::isSource(Value *V) {
  if (!isa<IntegerType>(V->getType()))
    return false;

  // TODO Allow zext to be sources.
  if (isa<Argument>(V))
    return true;
  else if (isa<LoadInst>(V))
    return true;
  else if (isa<BitCastInst>(V))
    return true;
  else if (auto *Call = dyn_cast<CallInst>(V))
    return Call->hasRetAttr(Attribute::AttrKind::ZExt);
  else if (auto *Trunc = dyn_cast<TruncInst>(V))
    return EqualTypeSize(Trunc);
  return false;
}

/// Return true if V will require any promoted values to be truncated for the
/// the IR to remain valid. We can't mutate the value type of these
/// instructions.
bool TypePromotion::isSink(Value *V) {
  // TODO The truncate also isn't actually necessary because we would already
  // proved that the data value is kept within the range of the original data
  // type.

  // Sinks are:
  // - points where the value in the register is being observed, such as an
  //   icmp, switch or store.
  // - points where value types have to match, such as calls and returns.
  // - zext are included to ease the transformation and are generally removed
  //   later on.
  if (auto *Store = dyn_cast<StoreInst>(V))
    return LessOrEqualTypeSize(Store->getValueOperand());
  if (auto *Return = dyn_cast<ReturnInst>(V))
    return LessOrEqualTypeSize(Return->getReturnValue());
  if (auto *ZExt = dyn_cast<ZExtInst>(V))
    return GreaterThanTypeSize(ZExt);
  if (auto *Switch = dyn_cast<SwitchInst>(V))
    return LessThanTypeSize(Switch->getCondition());
  if (auto *ICmp = dyn_cast<ICmpInst>(V))
    return ICmp->isSigned() || LessThanTypeSize(ICmp->getOperand(0));

  return isa<CallInst>(V);
}

/// Return whether this instruction can safely wrap.
bool TypePromotion::isSafeWrap(Instruction *I) {
  // We can support a, potentially, wrapping instruction (I) if:
  // - It is only used by an unsigned icmp.
  // - The icmp uses a constant.
  // - The wrapping value (I) is decreasing, i.e would underflow - wrapping
  //   around zero to become a larger number than before.
  // - The wrapping instruction (I) also uses a constant.
  //
  // We can then use the two constants to calculate whether the result would
  // wrap in respect to itself in the original bitwidth. If it doesn't wrap,
  // just underflows the range, the icmp would give the same result whether the
  // result has been truncated or not. We calculate this by:
  // - Zero extending both constants, if needed, to 32-bits.
  // - Take the absolute value of I's constant, adding this to the icmp const.
  // - Check that this value is not out of range for small type. If it is, it
  //   means that it has underflowed enough to wrap around the icmp constant.
  //
  // For example:
  //
  // %sub = sub i8 %a, 2
  // %cmp = icmp ule i8 %sub, 254
  //
  // If %a = 0, %sub = -2 == FE == 254
  // But if this is evalulated as a i32
  // %sub = -2 == FF FF FF FE == 4294967294
  // So the unsigned compares (i8 and i32) would not yield the same result.
  //
  // Another way to look at it is:
  // %a - 2 <= 254
  // %a + 2 <= 254 + 2
  // %a <= 256
  // And we can't represent 256 in the i8 format, so we don't support it.
  //
  // Whereas:
  //
  // %sub i8 %a, 1
  // %cmp = icmp ule i8 %sub, 254
  //
  // If %a = 0, %sub = -1 == FF == 255
  // As i32:
  // %sub = -1 == FF FF FF FF == 4294967295
  //
  // In this case, the unsigned compare results would be the same and this
  // would also be true for ult, uge and ugt:
  // - (255 < 254) == (0xFFFFFFFF < 254) == false
  // - (255 <= 254) == (0xFFFFFFFF <= 254) == false
  // - (255 > 254) == (0xFFFFFFFF > 254) == true
  // - (255 >= 254) == (0xFFFFFFFF >= 254) == true
  //
  // To demonstrate why we can't handle increasing values:
  //
  // %add = add i8 %a, 2
  // %cmp = icmp ult i8 %add, 127
  //
  // If %a = 254, %add = 256 == (i8 1)
  // As i32:
  // %add = 256
  //
  // (1 < 127) != (256 < 127)

  unsigned Opc = I->getOpcode();
  if (Opc != Instruction::Add && Opc != Instruction::Sub)
    return false;

  if (!I->hasOneUse() ||
      !isa<ICmpInst>(*I->user_begin()) ||
      !isa<ConstantInt>(I->getOperand(1)))
    return false;

  ConstantInt *OverflowConst = cast<ConstantInt>(I->getOperand(1));
  bool NegImm = OverflowConst->isNegative();
  bool IsDecreasing = ((Opc == Instruction::Sub) && !NegImm) ||
                       ((Opc == Instruction::Add) && NegImm);
  if (!IsDecreasing)
    return false;

  // Don't support an icmp that deals with sign bits.
  auto *CI = cast<ICmpInst>(*I->user_begin());
  if (CI->isSigned() || CI->isEquality())
    return false;

  ConstantInt *ICmpConst = nullptr;
  if (auto *Const = dyn_cast<ConstantInt>(CI->getOperand(0)))
    ICmpConst = Const;
  else if (auto *Const = dyn_cast<ConstantInt>(CI->getOperand(1)))
    ICmpConst = Const;
  else
    return false;

  // Now check that the result can't wrap on itself.
  APInt Total = ICmpConst->getValue().getBitWidth() < 32 ?
    ICmpConst->getValue().zext(32) : ICmpConst->getValue();

  Total += OverflowConst->getValue().getBitWidth() < 32 ?
    OverflowConst->getValue().abs().zext(32) : OverflowConst->getValue().abs();

  APInt Max = APInt::getAllOnesValue(TypePromotion::TypeSize);

  if (Total.getBitWidth() > Max.getBitWidth()) {
    if (Total.ugt(Max.zext(Total.getBitWidth())))
      return false;
  } else if (Max.getBitWidth() > Total.getBitWidth()) {
    if (Total.zext(Max.getBitWidth()).ugt(Max))
      return false;
  } else if (Total.ugt(Max))
    return false;

  LLVM_DEBUG(dbgs() << "IR Promotion: Allowing safe overflow for "
             << *I << "\n");
  SafeWrap.push_back(I);
  return true;
}

bool TypePromotion::shouldPromote(Value *V) {
  if (!isa<IntegerType>(V->getType()) || isSink(V))
    return false;

  if (isSource(V))
    return true;

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  if (isa<ICmpInst>(I))
    return false;

  return true;
}

/// Return whether we can safely mutate V's type to ExtTy without having to be
/// concerned with zero extending or truncation.
static bool isPromotedResultSafe(Value *V) {
  if (GenerateSignBits(V))
    return false;

  if (!isa<Instruction>(V))
    return true;

  if (!isa<OverflowingBinaryOperator>(V))
    return true;

  return cast<Instruction>(V)->hasNoUnsignedWrap();
}

void IRPromoter::ReplaceAllUsersOfWith(Value *From, Value *To) {
  SmallVector<Instruction*, 4> Users;
  Instruction *InstTo = dyn_cast<Instruction>(To);
  bool ReplacedAll = true;

  LLVM_DEBUG(dbgs() << "IR Promotion: Replacing " << *From << " with " << *To
             << "\n");

  for (Use &U : From->uses()) {
    auto *User = cast<Instruction>(U.getUser());
    if (InstTo && User->isIdenticalTo(InstTo)) {
      ReplacedAll = false;
      continue;
    }
    Users.push_back(User);
  }

  for (auto *U : Users)
    U->replaceUsesOfWith(From, To);

  if (ReplacedAll)
    if (auto *I = dyn_cast<Instruction>(From))
      InstsToRemove.insert(I);
}

void IRPromoter::PrepareWrappingAdds() {
  LLVM_DEBUG(dbgs() << "IR Promotion: Prepare wrapping adds.\n");
  IRBuilder<> Builder{Ctx};

  // For adds that safely wrap and use a negative immediate as operand 1, we
  // create an equivalent instruction using a positive immediate.
  // That positive immediate can then be zext along with all the other
  // immediates later.
  for (auto *I : SafeWrap) {
    if (I->getOpcode() != Instruction::Add)
      continue;

    LLVM_DEBUG(dbgs() << "IR Promotion: Adjusting " << *I << "\n");
    assert((isa<ConstantInt>(I->getOperand(1)) &&
            cast<ConstantInt>(I->getOperand(1))->isNegative()) &&
           "Wrapping should have a negative immediate as the second operand");

    auto Const = cast<ConstantInt>(I->getOperand(1));
    auto *NewConst = ConstantInt::get(Ctx, Const->getValue().abs());
    Builder.SetInsertPoint(I);
    Value *NewVal = Builder.CreateSub(I->getOperand(0), NewConst);
    if (auto *NewInst = dyn_cast<Instruction>(NewVal)) {
      NewInst->copyIRFlags(I);
      NewInsts.insert(NewInst);
    }
    InstsToRemove.insert(I);
    I->replaceAllUsesWith(NewVal);
    LLVM_DEBUG(dbgs() << "IR Promotion: New equivalent: " << *NewVal << "\n");
  }
  for (auto *I : NewInsts)
    Visited.insert(I);
}

void IRPromoter::ExtendSources() {
  IRBuilder<> Builder{Ctx};

  auto InsertZExt = [&](Value *V, Instruction *InsertPt) {
    assert(V->getType() != ExtTy && "zext already extends to i32");
    LLVM_DEBUG(dbgs() << "IR Promotion: Inserting ZExt for " << *V << "\n");
    Builder.SetInsertPoint(InsertPt);
    if (auto *I = dyn_cast<Instruction>(V))
      Builder.SetCurrentDebugLocation(I->getDebugLoc());

    Value *ZExt = Builder.CreateZExt(V, ExtTy);
    if (auto *I = dyn_cast<Instruction>(ZExt)) {
      if (isa<Argument>(V))
        I->moveBefore(InsertPt);
      else
        I->moveAfter(InsertPt);
      NewInsts.insert(I);
    }

    ReplaceAllUsersOfWith(V, ZExt);
  };

  // Now, insert extending instructions between the sources and their users.
  LLVM_DEBUG(dbgs() << "IR Promotion: Promoting sources:\n");
  for (auto V : Sources) {
    LLVM_DEBUG(dbgs() << " - " << *V << "\n");
    if (auto *I = dyn_cast<Instruction>(V))
      InsertZExt(I, I);
    else if (auto *Arg = dyn_cast<Argument>(V)) {
      BasicBlock &BB = Arg->getParent()->front();
      InsertZExt(Arg, &*BB.getFirstInsertionPt());
    } else {
      llvm_unreachable("unhandled source that needs extending");
    }
    Promoted.insert(V);
  }
}

void IRPromoter::PromoteTree() {
  LLVM_DEBUG(dbgs() << "IR Promotion: Mutating the tree..\n");

  IRBuilder<> Builder{Ctx};

  // Mutate the types of the instructions within the tree. Here we handle
  // constant operands.
  for (auto *V : Visited) {
    if (Sources.count(V))
      continue;

    auto *I = cast<Instruction>(V);
    if (Sinks.count(I))
      continue;

    for (unsigned i = 0, e = I->getNumOperands(); i < e; ++i) {
      Value *Op = I->getOperand(i);
      if ((Op->getType() == ExtTy) || !isa<IntegerType>(Op->getType()))
        continue;

      if (auto *Const = dyn_cast<ConstantInt>(Op)) {
        Constant *NewConst = ConstantExpr::getZExt(Const, ExtTy);
        I->setOperand(i, NewConst);
      } else if (isa<UndefValue>(Op))
        I->setOperand(i, UndefValue::get(ExtTy));
    }

    // Mutate the result type, unless this is an icmp.
    if (!isa<ICmpInst>(I)) {
      I->mutateType(ExtTy);
      Promoted.insert(I);
    }
  }
}

void IRPromoter::TruncateSinks() {
  LLVM_DEBUG(dbgs() << "IR Promotion: Fixing up the sinks:\n");

  IRBuilder<> Builder{Ctx};

  auto InsertTrunc = [&](Value *V, Type *TruncTy) -> Instruction* {
    if (!isa<Instruction>(V) || !isa<IntegerType>(V->getType()))
      return nullptr;

    if ((!Promoted.count(V) && !NewInsts.count(V)) || Sources.count(V))
      return nullptr;

    LLVM_DEBUG(dbgs() << "IR Promotion: Creating " << *TruncTy << " Trunc for "
               << *V << "\n");
    Builder.SetInsertPoint(cast<Instruction>(V));
    auto *Trunc = dyn_cast<Instruction>(Builder.CreateTrunc(V, TruncTy));
    if (Trunc)
      NewInsts.insert(Trunc);
    return Trunc;
  };

  // Fix up any stores or returns that use the results of the promoted
  // chain.
  for (auto I : Sinks) {
    LLVM_DEBUG(dbgs() << "IR Promotion: For Sink: " << *I << "\n");

    // Handle calls separately as we need to iterate over arg operands.
    if (auto *Call = dyn_cast<CallInst>(I)) {
      for (unsigned i = 0; i < Call->getNumArgOperands(); ++i) {
        Value *Arg = Call->getArgOperand(i);
        Type *Ty = TruncTysMap[Call][i];
        if (Instruction *Trunc = InsertTrunc(Arg, Ty)) {
          Trunc->moveBefore(Call);
          Call->setArgOperand(i, Trunc);
        }
      }
      continue;
    }

    // Special case switches because we need to truncate the condition.
    if (auto *Switch = dyn_cast<SwitchInst>(I)) {
      Type *Ty = TruncTysMap[Switch][0];
      if (Instruction *Trunc = InsertTrunc(Switch->getCondition(), Ty)) {
        Trunc->moveBefore(Switch);
        Switch->setCondition(Trunc);
      }
      continue;
    }

    // Now handle the others.
    for (unsigned i = 0; i < I->getNumOperands(); ++i) {
      Type *Ty = TruncTysMap[I][i];
      if (Instruction *Trunc = InsertTrunc(I->getOperand(i), Ty)) {
        Trunc->moveBefore(I);
        I->setOperand(i, Trunc);
      }
    }
  }
}

void IRPromoter::Cleanup() {
  LLVM_DEBUG(dbgs() << "IR Promotion: Cleanup..\n");
  // Some zexts will now have become redundant, along with their trunc
  // operands, so remove them
  for (auto V : Visited) {
    if (!isa<ZExtInst>(V))
      continue;

    auto ZExt = cast<ZExtInst>(V);
    if (ZExt->getDestTy() != ExtTy)
      continue;

    Value *Src = ZExt->getOperand(0);
    if (ZExt->getSrcTy() == ZExt->getDestTy()) {
      LLVM_DEBUG(dbgs() << "IR Promotion: Removing unnecessary cast: " << *ZExt
                 << "\n");
      ReplaceAllUsersOfWith(ZExt, Src);
      continue;
    }

    // Unless they produce a value that is narrower than ExtTy, we can
    // replace the result of the zext with the input of a newly inserted
    // trunc.
    if (NewInsts.count(Src) && isa<TruncInst>(Src) &&
        Src->getType() == OrigTy) {
      auto *Trunc = cast<TruncInst>(Src);
      assert(Trunc->getOperand(0)->getType() == ExtTy &&
             "expected inserted trunc to be operating on i32");
      ReplaceAllUsersOfWith(ZExt, Trunc->getOperand(0));
    }
  }

  for (auto *I : InstsToRemove) {
    LLVM_DEBUG(dbgs() << "IR Promotion: Removing " << *I << "\n");
    I->dropAllReferences();
    I->eraseFromParent();
  }
}

void IRPromoter::ConvertTruncs() {
  LLVM_DEBUG(dbgs() << "IR Promotion: Converting truncs..\n");
  IRBuilder<> Builder{Ctx};

  for (auto *V : Visited) {
    if (!isa<TruncInst>(V) || Sources.count(V))
      continue;

    auto *Trunc = cast<TruncInst>(V);
    Builder.SetInsertPoint(Trunc);
    IntegerType *SrcTy = cast<IntegerType>(Trunc->getOperand(0)->getType());
    IntegerType *DestTy = cast<IntegerType>(TruncTysMap[Trunc][0]);

    unsigned NumBits = DestTy->getScalarSizeInBits();
    ConstantInt *Mask =
      ConstantInt::get(SrcTy, APInt::getMaxValue(NumBits).getZExtValue());
    Value *Masked = Builder.CreateAnd(Trunc->getOperand(0), Mask);

    if (auto *I = dyn_cast<Instruction>(Masked))
      NewInsts.insert(I);

    ReplaceAllUsersOfWith(Trunc, Masked);
  }
}

void IRPromoter::Mutate() {
  LLVM_DEBUG(dbgs() << "IR Promotion: Promoting use-def chains from "
             << OrigTy->getBitWidth() << " to " << PromotedWidth << "-bits\n");

  // Cache original types of the values that will likely need truncating
  for (auto *I : Sinks) {
    if (auto *Call = dyn_cast<CallInst>(I)) {
      for (unsigned i = 0; i < Call->getNumArgOperands(); ++i) {
        Value *Arg = Call->getArgOperand(i);
        TruncTysMap[Call].push_back(Arg->getType());
      }
    } else if (auto *Switch = dyn_cast<SwitchInst>(I))
      TruncTysMap[I].push_back(Switch->getCondition()->getType());
    else {
      for (unsigned i = 0; i < I->getNumOperands(); ++i)
        TruncTysMap[I].push_back(I->getOperand(i)->getType());
    }
  }
  for (auto *V : Visited) {
    if (!isa<TruncInst>(V) || Sources.count(V))
      continue;
    auto *Trunc = cast<TruncInst>(V);
    TruncTysMap[Trunc].push_back(Trunc->getDestTy());
  }

  // Convert adds using negative immediates to equivalent instructions that use
  // positive constants.
  PrepareWrappingAdds();

  // Insert zext instructions between sources and their users.
  ExtendSources();

  // Promote visited instructions, mutating their types in place.
  PromoteTree();

  // Convert any truncs, that aren't sources, into AND masks.
  ConvertTruncs();

  // Insert trunc instructions for use by calls, stores etc...
  TruncateSinks();

  // Finally, remove unecessary zexts and truncs, delete old instructions and
  // clear the data structures.
  Cleanup();

  LLVM_DEBUG(dbgs() << "IR Promotion: Mutation complete\n");
}

/// We disallow booleans to make life easier when dealing with icmps but allow
/// any other integer that fits in a scalar register. Void types are accepted
/// so we can handle switches.
bool TypePromotion::isSupportedType(Value *V) {
  Type *Ty = V->getType();

  // Allow voids and pointers, these won't be promoted.
  if (Ty->isVoidTy() || Ty->isPointerTy())
    return true;

  if (!isa<IntegerType>(Ty) ||
      cast<IntegerType>(Ty)->getBitWidth() == 1 ||
      cast<IntegerType>(Ty)->getBitWidth() > RegisterBitWidth)
    return false;

  return LessOrEqualTypeSize(V);
}

/// We accept most instructions, as well as Arguments and ConstantInsts. We
/// Disallow casts other than zext and truncs and only allow calls if their
/// return value is zeroext. We don't allow opcodes that can introduce sign
/// bits.
bool TypePromotion::isSupportedValue(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    switch (I->getOpcode()) {
    default:
      return isa<BinaryOperator>(I) && isSupportedType(I) &&
             !GenerateSignBits(I);
    case Instruction::GetElementPtr:
    case Instruction::Store:
    case Instruction::Br:
    case Instruction::Switch:
      return true;
    case Instruction::PHI:
    case Instruction::Select:
    case Instruction::Ret:
    case Instruction::Load:
    case Instruction::Trunc:
    case Instruction::BitCast:
      return isSupportedType(I);
    case Instruction::ZExt:
      return isSupportedType(I->getOperand(0));
    case Instruction::ICmp:
      // Now that we allow small types than TypeSize, only allow icmp of
      // TypeSize because they will require a trunc to be legalised.
      // TODO: Allow icmp of smaller types, and calculate at the end
      // whether the transform would be beneficial.
      if (isa<PointerType>(I->getOperand(0)->getType()))
        return true;
      return EqualTypeSize(I->getOperand(0));
    case Instruction::Call: {
      // Special cases for calls as we need to check for zeroext
      // TODO We should accept calls even if they don't have zeroext, as they
      // can still be sinks.
      auto *Call = cast<CallInst>(I);
      return isSupportedType(Call) &&
             Call->hasRetAttr(Attribute::AttrKind::ZExt);
    }
    }
  } else if (isa<Constant>(V) && !isa<ConstantExpr>(V)) {
    return isSupportedType(V);
  } else if (isa<Argument>(V))
    return isSupportedType(V);

  return isa<BasicBlock>(V);
}

/// Check that the type of V would be promoted and that the original type is
/// smaller than the targeted promoted type. Check that we're not trying to
/// promote something larger than our base 'TypeSize' type.
bool TypePromotion::isLegalToPromote(Value *V) {

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return true;

  if (SafeToPromote.count(I))
   return true;

  if (isPromotedResultSafe(V) || isSafeWrap(I)) {
    SafeToPromote.insert(I);
    return true;
  }
  return false;
}

bool TypePromotion::TryToPromote(Value *V, unsigned PromotedWidth) {
  Type *OrigTy = V->getType();
  TypeSize = OrigTy->getPrimitiveSizeInBits();
  SafeToPromote.clear();
  SafeWrap.clear();

  if (!isSupportedValue(V) || !shouldPromote(V) || !isLegalToPromote(V))
    return false;

  LLVM_DEBUG(dbgs() << "IR Promotion: TryToPromote: " << *V << ", from "
             << TypeSize << " bits to " << PromotedWidth << "\n");

  SetVector<Value*> WorkList;
  SetVector<Value*> Sources;
  SetVector<Instruction*> Sinks;
  SetVector<Value*> CurrentVisited;
  WorkList.insert(V);

  // Return true if V was added to the worklist as a supported instruction,
  // if it was already visited, or if we don't need to explore it (e.g.
  // pointer values and GEPs), and false otherwise.
  auto AddLegalInst = [&](Value *V) {
    if (CurrentVisited.count(V))
      return true;

    // Ignore GEPs because they don't need promoting and the constant indices
    // will prevent the transformation.
    if (isa<GetElementPtrInst>(V))
      return true;

    if (!isSupportedValue(V) || (shouldPromote(V) && !isLegalToPromote(V))) {
      LLVM_DEBUG(dbgs() << "IR Promotion: Can't handle: " << *V << "\n");
      return false;
    }

    WorkList.insert(V);
    return true;
  };

  // Iterate through, and add to, a tree of operands and users in the use-def.
  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    if (CurrentVisited.count(V))
      continue;

    // Ignore non-instructions, other than arguments.
    if (!isa<Instruction>(V) && !isSource(V))
      continue;

    // If we've already visited this value from somewhere, bail now because
    // the tree has already been explored.
    // TODO: This could limit the transform, ie if we try to promote something
    // from an i8 and fail first, before trying an i16.
    if (AllVisited.count(V))
      return false;

    CurrentVisited.insert(V);
    AllVisited.insert(V);

    // Calls can be both sources and sinks.
    if (isSink(V))
      Sinks.insert(cast<Instruction>(V));

    if (isSource(V))
      Sources.insert(V);

    if (!isSink(V) && !isSource(V)) {
      if (auto *I = dyn_cast<Instruction>(V)) {
        // Visit operands of any instruction visited.
        for (auto &U : I->operands()) {
          if (!AddLegalInst(U))
            return false;
        }
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

  LLVM_DEBUG(dbgs() << "IR Promotion: Visited nodes:\n";
             for (auto *I : CurrentVisited)
               I->dump();
             );

  unsigned ToPromote = 0;
  unsigned NonFreeArgs = 0;
  SmallPtrSet<BasicBlock*, 4> Blocks;
  for (auto *V : CurrentVisited) {
    if (auto *I = dyn_cast<Instruction>(V))
      Blocks.insert(I->getParent());

    if (Sources.count(V)) {
      if (auto *Arg = dyn_cast<Argument>(V))
        if (!Arg->hasZExtAttr() && !Arg->hasSExtAttr())
          ++NonFreeArgs;
      continue;
    }

    if (Sinks.count(cast<Instruction>(V)))
      continue;
     ++ToPromote;
   }

  // DAG optimizations should be able to handle these cases better, especially
  // for function arguments.
  if (ToPromote < 2 || (Blocks.size() == 1 && (NonFreeArgs > SafeWrap.size())))
    return false;

  if (ToPromote < 2)
    return false;

  IRPromoter Promoter(*Ctx, cast<IntegerType>(OrigTy), PromotedWidth,
                      CurrentVisited, Sources, Sinks, SafeWrap);
  Promoter.Mutate();
  return true;
}

bool TypePromotion::runOnFunction(Function &F) {
  if (skipFunction(F) || DisablePromotion)
    return false;

  LLVM_DEBUG(dbgs() << "IR Promotion: Running on " << F.getName() << "\n");

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  AllVisited.clear();
  SafeToPromote.clear();
  SafeWrap.clear();
  bool MadeChange = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  const TargetMachine &TM = TPC->getTM<TargetMachine>();
  const TargetSubtargetInfo *SubtargetInfo = TM.getSubtargetImpl(F);
  const TargetLowering *TLI = SubtargetInfo->getTargetLowering();
  const TargetTransformInfo &TII =
    getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  RegisterBitWidth = TII.getRegisterBitWidth(false);
  Ctx = &F.getParent()->getContext();

  // Search up from icmps to try to promote their operands.
  for (BasicBlock &BB : F) {
    for (auto &I : BB) {
      if (AllVisited.count(&I))
        continue;

      if (!isa<ICmpInst>(&I))
        continue;

      auto *ICmp = cast<ICmpInst>(&I);
      // Skip signed or pointer compares
      if (ICmp->isSigned() ||
          !isa<IntegerType>(ICmp->getOperand(0)->getType()))
        continue;

      LLVM_DEBUG(dbgs() << "IR Promotion: Searching from: " << *ICmp << "\n");

      for (auto &Op : ICmp->operands()) {
        if (auto *I = dyn_cast<Instruction>(Op)) {
          EVT SrcVT = TLI->getValueType(DL, I->getType());
          if (SrcVT.isSimple() && TLI->isTypeLegal(SrcVT.getSimpleVT()))
            break;

          if (TLI->getTypeAction(ICmp->getContext(), SrcVT) !=
              TargetLowering::TypePromoteInteger)
            break;

          EVT PromotedVT = TLI->getTypeToTransformTo(ICmp->getContext(), SrcVT);
          if (RegisterBitWidth < PromotedVT.getSizeInBits()) {
            LLVM_DEBUG(dbgs() << "IR Promotion: Couldn't find target register "
                       << "for promoted type\n");
            break;
          }

          MadeChange |= TryToPromote(I, PromotedVT.getSizeInBits());
          break;
        }
      }
    }
    LLVM_DEBUG(if (verifyFunction(F, &dbgs())) {
                dbgs() << F;
                report_fatal_error("Broken function after type promotion");
               });
  }
  if (MadeChange)
    LLVM_DEBUG(dbgs() << "After TypePromotion: " << F << "\n");

  AllVisited.clear();
  SafeToPromote.clear();
  SafeWrap.clear();

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(TypePromotion, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_END(TypePromotion, DEBUG_TYPE, PASS_NAME, false, false)

char TypePromotion::ID = 0;

FunctionPass *llvm::createTypePromotionPass() {
  return new TypePromotion();
}
