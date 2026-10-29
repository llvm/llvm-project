//===- AIRLegalizeUnsupportedIR.cpp - Strip unsupported IR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRLegalizeUnsupportedIR.h"
#include "AIR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/AIR.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/Utils/Local.h"
#include <optional>
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Transforms/Utils/LowerAtomic.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "air-legalize-unsupported-ir"

namespace {

void lowerCmpIntrinsics(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Intrinsic::ID ID = CI->getIntrinsicID();
          if (ID == Intrinsic::scmp || ID == Intrinsic::ucmp)
            Calls.push_back(CI);
        }
    for (CallInst *CI : Calls) {
      bool Signed = CI->getIntrinsicID() == Intrinsic::scmp;
      IRBuilder<> B(CI);
      Value *A = CI->getArgOperand(0);
      Value *Bv = CI->getArgOperand(1);
      Value *Gt = Signed ? B.CreateICmpSGT(A, Bv) : B.CreateICmpUGT(A, Bv);
      Value *Lt = Signed ? B.CreateICmpSLT(A, Bv) : B.CreateICmpULT(A, Bv);
      Type *RetTy = CI->getType();
      Value *Res =
          B.CreateSub(B.CreateZExt(Gt, RetTy), B.CreateZExt(Lt, RetTy));
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
    }
  }

  for (auto It = M.begin(); It != M.end();) {
    Function &F = *It++;
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::scmp || ID == Intrinsic::ucmp) && F.use_empty())
      F.eraseFromParent();
  }
}

// AIR integer functions on 65- to 128-bit integers back to the LLVM
// intrinsics they came from, which expandWideIntegers splits into limbs.
// The portable passes name an intrinsic's AIR function after its type
// (air.clz.i128, air.abs.s.i128: in every module ZLUDA links its i128
// helpers into), and the AIR writer has no wide integer calls.
void wideAIRFunctionsToIntrinsics(Module &M) {
  static constexpr std::pair<StringLiteral, Intrinsic::ID> Fns[] = {
      {"air.clz.", Intrinsic::ctlz},         {"air.ctz.", Intrinsic::cttz},
      {"air.popcount.", Intrinsic::ctpop},   {"air.abs.s.", Intrinsic::abs},
      {"air.min.s.", Intrinsic::smin},       {"air.max.s.", Intrinsic::smax},
      {"air.min.u.", Intrinsic::umin},       {"air.max.u.", Intrinsic::umax},
      {"air.reverse_bits.", Intrinsic::bitreverse}};
  for (Function &F : make_early_inc_range(M)) {
    if (!F.isDeclaration())
      continue;
    Type *T = F.getReturnType();
    if (!T->isIntegerTy() || T->getIntegerBitWidth() <= 64)
      continue;
    const auto *Fn = find_if(Fns, [&](const auto &P) {
      return F.getName().starts_with(P.first);
    });
    if (Fn == std::end(Fns))
      continue;
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &F)
        continue;
      IRBuilder<> B(CI);
      SmallVector<Value *, 2> Args(CI->args());
      // llvm.abs takes the INT_MIN-is-poison flag AIR's abs has not.
      if (Fn->second == Intrinsic::abs)
        Args.push_back(B.getFalse());
      Value *R = B.CreateIntrinsic(Fn->second, {T}, Args);
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
}

// Integers of 65 to 128 bits (AIR has none; agx-raise-to-air keeps them as
// a type hint, llvm/IR/AIR.h) as two i64 limbs. The high limb of an N-bit
// value holds its bits 64 to N - 1, zero above.
void expandWideIntegers(Module &M) {
  auto isWide = [](Type *T) {
    return T->isIntegerTy() && T->getIntegerBitWidth() > 64 &&
           T->getIntegerBitWidth() <= 128;
  };
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  using Limbs2 = std::pair<Value *, Value *>;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    DenseMap<Value *, Limbs2> Limbs;
    SmallVector<Instruction *, 8> Wide;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isWide(I.getType()) || llvm::any_of(I.operands(), [&](Value *Op) {
              return isWide(Op->getType());
            }))
          Wide.push_back(&I);
    if (Wide.empty())
      continue;
    auto C64 = [&](uint64_t V) { return ConstantInt::get(I64, V); };
    auto umulh = [&](IRBuilder<> &B, Value *A, Value *Bv) -> Value * {
      Value *Mask = C64(0xffffffffull);
      Value *AL = B.CreateAnd(A, Mask), *AH = B.CreateLShr(A, 32);
      Value *BL = B.CreateAnd(Bv, Mask), *BH = B.CreateLShr(Bv, 32);
      Value *LL = B.CreateMul(AL, BL);
      Value *LH = B.CreateMul(AL, BH);
      Value *HL = B.CreateMul(AH, BL);
      Value *HH = B.CreateMul(AH, BH);
      Value *Mid =
          B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)),
                      B.CreateAnd(HL, Mask));
      return B.CreateAdd(B.CreateAdd(HH, B.CreateAdd(B.CreateLShr(LH, 32),
                                                     B.CreateLShr(HL, 32))),
                         B.CreateLShr(Mid, 32));
    };
    auto limbsOf = [&](Value *V) -> Limbs2 {
      auto It = Limbs.find(V);
      if (It != Limbs.end())
        return It->second;
      if (auto *C = dyn_cast<ConstantInt>(V)) {
        APInt A = C->getValue().zext(128);
        return {ConstantInt::get(I64, A.trunc(64)),
                ConstantInt::get(I64, A.lshr(64).trunc(64))};
      }
      if (isa<UndefValue>(V))
        return {PoisonValue::get(I64), PoisonValue::get(I64)};
      report_fatal_error("AIRWriter: unmapped wide integer operand");
    };
    // The high limb of a value of W bits: zero above bit W - 64
    // (canonical), or sign-extended from it.
    auto canon = [&](IRBuilder<> &B, Value *H, unsigned W) -> Value * {
      return W == 128 ? H : B.CreateAnd(H, C64(~0ull >> (128 - W)));
    };
    auto signedHi = [&](IRBuilder<> &B, Value *H, unsigned W) -> Value * {
      return W == 128 ? H : B.CreateAShr(B.CreateShl(H, 128 - W), 128 - W);
    };
    // Shifts of (L, H) by S (an i64 below 128).
    auto shift = [&](IRBuilder<> &B, Instruction::BinaryOps Op, Value *L,
                     Value *H, Value *S) -> Limbs2 {
      Value *Big = B.CreateICmpUGE(S, C64(64));
      Value *Zero = B.CreateICmpEQ(S, C64(0));
      Value *S63 = B.CreateAnd(S, C64(63));
      Value *Inv = B.CreateAnd(B.CreateSub(C64(64), S63), C64(63));
      if (Op == Instruction::Shl) {
        Value *Lo = B.CreateShl(L, S63);
        Value *Carry = B.CreateSelect(Zero, C64(0), B.CreateLShr(L, Inv));
        Value *Hi = B.CreateOr(B.CreateShl(H, S63), Carry);
        return {B.CreateSelect(Big, C64(0), Lo),
                B.CreateSelect(Big, B.CreateShl(L, S63), Hi)};
      }
      bool Arith = Op == Instruction::AShr;
      Value *Fill = Arith ? B.CreateAShr(H, 63) : C64(0);
      Value *Hi = Arith ? B.CreateAShr(H, S63) : B.CreateLShr(H, S63);
      Value *Carry = B.CreateSelect(Zero, C64(0), B.CreateShl(H, Inv));
      Value *Lo = B.CreateOr(B.CreateLShr(L, S63), Carry);
      return {B.CreateSelect(Big, Arith ? B.CreateAShr(H, S63)
                                        : B.CreateLShr(H, S63),
                             Lo),
              B.CreateSelect(Big, Fill, Hi)};
    };
    auto icmp = [&](IRBuilder<> &B, CmpInst::Predicate P, Limbs2 X, Limbs2 Y,
                    unsigned W) -> Value * {
      auto [L1, H1] = X;
      auto [L2, H2] = Y;
      if (P == CmpInst::ICMP_EQ || P == CmpInst::ICMP_NE) {
        Value *Eq = B.CreateAnd(B.CreateICmpEQ(L1, L2), B.CreateICmpEQ(H1, H2));
        return P == CmpInst::ICMP_EQ ? Eq : B.CreateNot(Eq);
      }
      if (ICmpInst::isSigned(P)) {
        H1 = signedHi(B, H1, W);
        H2 = signedHi(B, H2, W);
      }
      // The high limbs decide unless equal; then the low ones, unsigned.
      Value *HiCmp = B.CreateICmp(ICmpInst::getStrictPredicate(P), H1, H2);
      Value *LoCmp = B.CreateICmp(
          ICmpInst::getUnsignedPredicate(P), L1, L2);
      return B.CreateSelect(B.CreateICmpEQ(H1, H2), LoCmp, HiCmp);
    };
    SmallVector<std::pair<PHINode *, std::pair<PHINode *, PHINode *>>, 4> Phis;
    for (Instruction *I : Wide) {
      IRBuilder<> B(I);
      unsigned W = isWide(I->getType()) ? I->getType()->getIntegerBitWidth()
                                        : 0;
      auto Set = [&](Value *Lo, Value *Hi) {
        Limbs[I] = {Lo, canon(B, Hi, W)};
      };
      if (auto *ZE = dyn_cast<ZExtInst>(I)) {
        Value *X = ZE->getOperand(0);
        if (isWide(X->getType()))
          Limbs[I] = limbsOf(X);
        else
          Set(B.CreateZExtOrTrunc(X, I64), C64(0));
      } else if (auto *SE2 = dyn_cast<SExtInst>(I)) {
        Value *X = SE2->getOperand(0);
        if (isWide(X->getType())) {
          auto [L, H] = limbsOf(X);
          Set(L, signedHi(B, H, X->getType()->getIntegerBitWidth()));
        } else {
          Value *Lo = B.CreateSExtOrTrunc(X, I64);
          Set(Lo, B.CreateAShr(Lo, 63));
        }
      } else if (auto *TR = dyn_cast<TruncInst>(I)) {
        auto [Lo, Hi] = limbsOf(TR->getOperand(0));
        if (W)
          Set(Lo, Hi);
        else
          TR->replaceAllUsesWith(B.CreateZExtOrTrunc(Lo, TR->getType()));
      } else if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        auto [L1, H1] = limbsOf(BO->getOperand(0));
        auto [L2, H2] = limbsOf(BO->getOperand(1));
        switch (BO->getOpcode()) {
        case Instruction::Mul:
          Set(B.CreateMul(L1, L2),
              B.CreateAdd(umulh(B, L1, L2), B.CreateAdd(B.CreateMul(L1, H2),
                                                        B.CreateMul(H1, L2))));
          break;
        case Instruction::Add: {
          Value *Lo = B.CreateAdd(L1, L2);
          Value *Carry = B.CreateZExt(B.CreateICmpULT(Lo, L1), I64);
          Set(Lo, B.CreateAdd(B.CreateAdd(H1, H2), Carry));
          break;
        }
        case Instruction::Sub: {
          Value *Borrow = B.CreateZExt(B.CreateICmpULT(L1, L2), I64);
          Set(B.CreateSub(L1, L2), B.CreateSub(B.CreateSub(H1, H2), Borrow));
          break;
        }
        case Instruction::And:
          Set(B.CreateAnd(L1, L2), B.CreateAnd(H1, H2));
          break;
        case Instruction::Or:
          Set(B.CreateOr(L1, L2), B.CreateOr(H1, H2));
          break;
        case Instruction::Xor:
          Set(B.CreateXor(L1, L2), B.CreateXor(H1, H2));
          break;
        case Instruction::Shl:
        case Instruction::LShr:
        case Instruction::AShr: {
          // Amounts of W or more are poison: any result does.
          Value *H = BO->getOpcode() == Instruction::AShr
                         ? signedHi(B, H1, W)
                         : H1;
          auto [Lo, Hi] = shift(B, BO->getOpcode(), L1, H,
                                B.CreateAnd(L2, C64(127)));
          Set(Lo, Hi);
          break;
        }
        default:
          report_fatal_error(Twine("AIRWriter: unhandled wide integer op '") +
                             BO->getOpcodeName() + "'");
        }
      } else if (auto *IC = dyn_cast<ICmpInst>(I)) {
        unsigned OW = IC->getOperand(0)->getType()->getIntegerBitWidth();
        IC->replaceAllUsesWith(icmp(B, IC->getPredicate(),
                                    limbsOf(IC->getOperand(0)),
                                    limbsOf(IC->getOperand(1)), OW));
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        if (Sel->getCondition()->getType()->isVectorTy())
          report_fatal_error("AIRWriter: wide integer vector select");
        auto [L1, H1] = limbsOf(Sel->getTrueValue());
        auto [L2, H2] = limbsOf(Sel->getFalseValue());
        Value *Cond = Sel->getCondition();
        Limbs[I] = {B.CreateSelect(Cond, L1, L2), B.CreateSelect(Cond, H1, H2)};
      } else if (auto *Phi = dyn_cast<PHINode>(I)) {
        PHINode *Lo = B.CreatePHI(I64, Phi->getNumIncomingValues());
        PHINode *Hi = B.CreatePHI(I64, Phi->getNumIncomingValues());
        Limbs[I] = {Lo, Hi};
        Phis.push_back({Phi, {Lo, Hi}});
      } else if (auto *Fr = dyn_cast<FreezeInst>(I)) {
        auto [L, H] = limbsOf(Fr->getOperand(0));
        Limbs[I] = {B.CreateFreeze(L), B.CreateFreeze(H)};
      } else if (auto *LI = dyn_cast<LoadInst>(I); LI && LI->isSimple() && W) {
        // The low eight bytes, then the rest (little endian).
        Value *P = LI->getPointerOperand();
        Align A = LI->getAlign();
        Type *HiTy = Type::getIntNTy(Ctx, W - 64);
        if (W - 64 != 8 && W - 64 != 16 && W - 64 != 32 && W - 64 != 64)
          report_fatal_error("AIRWriter: wide integer load of an odd width");
        Value *Lo = B.CreateAlignedLoad(I64, P, A, LI->isVolatile());
        Value *HP = B.CreateConstInBoundsGEP1_64(Type::getInt8Ty(Ctx), P, 8);
        Value *Hi = B.CreateAlignedLoad(HiTy, HP, commonAlignment(A, 8),
                                        LI->isVolatile());
        Limbs[I] = {Lo, B.CreateZExt(Hi, I64)};
      } else if (auto *SI = dyn_cast<StoreInst>(I);
                 SI && SI->isSimple() && isWide(SI->getValueOperand()->getType())) {
        unsigned SW = SI->getValueOperand()->getType()->getIntegerBitWidth();
        if (SW - 64 != 8 && SW - 64 != 16 && SW - 64 != 32 && SW - 64 != 64)
          report_fatal_error("AIRWriter: wide integer store of an odd width");
        auto [L, H] = limbsOf(SI->getValueOperand());
        Value *P = SI->getPointerOperand();
        Align A = SI->getAlign();
        B.CreateAlignedStore(L, P, A, SI->isVolatile());
        Value *HP = B.CreateConstInBoundsGEP1_64(Type::getInt8Ty(Ctx), P, 8);
        B.CreateAlignedStore(B.CreateTrunc(H, Type::getIntNTy(Ctx, SW - 64)), HP,
                             commonAlignment(A, 8), SI->isVolatile());
      } else if (auto *BC = dyn_cast<BitCastInst>(I)) {
        // To or from a vector of 128 bits, through <2 x i64>.
        auto *V2 = FixedVectorType::get(I64, 2);
        if (W == 128) {
          Value *V = B.CreateBitCast(BC->getOperand(0), V2);
          Limbs[I] = {B.CreateExtractElement(V, uint64_t(0)),
                      B.CreateExtractElement(V, uint64_t(1))};
        } else if (BC->getOperand(0)->getType()->getIntegerBitWidth() == 128) {
          auto [L, H] = limbsOf(BC->getOperand(0));
          Value *V = B.CreateInsertElement(PoisonValue::get(V2), L, uint64_t(0));
          V = B.CreateInsertElement(V, H, uint64_t(1));
          BC->replaceAllUsesWith(B.CreateBitCast(V, BC->getType()));
        } else {
          report_fatal_error("AIRWriter: wide integer bitcast");
        }
      } else if (auto *II = dyn_cast<IntrinsicInst>(I); II && W) {
        auto [L1, H1] = limbsOf(II->getArgOperand(0));
        switch (II->getIntrinsicID()) {
        case Intrinsic::abs: {
          Value *Neg = B.CreateICmpSLT(signedHi(B, H1, W), C64(0));
          Value *NL = B.CreateNeg(L1);
          Value *NH = B.CreateSub(B.CreateNeg(H1),
                                  B.CreateZExt(B.CreateICmpNE(L1, C64(0)), I64));
          Set(B.CreateSelect(Neg, NL, L1), B.CreateSelect(Neg, NH, H1));
          break;
        }
        case Intrinsic::ctlz: {
          // Leading zeros of the W bits.
          Value *ZH = B.CreateBinaryIntrinsic(Intrinsic::ctlz, H1, B.getFalse());
          Value *ZL = B.CreateBinaryIntrinsic(Intrinsic::ctlz, L1, B.getFalse());
          Value *R = B.CreateSelect(B.CreateICmpEQ(H1, C64(0)),
                                    B.CreateAdd(ZL, C64(64)), ZH);
          Set(B.CreateSub(R, C64(128 - W)), C64(0));
          break;
        }
        case Intrinsic::cttz: {
          Value *TH = B.CreateBinaryIntrinsic(Intrinsic::cttz, H1, B.getFalse());
          Value *TL = B.CreateBinaryIntrinsic(Intrinsic::cttz, L1, B.getFalse());
          Value *R = B.CreateSelect(B.CreateICmpEQ(L1, C64(0)),
                                    B.CreateAdd(TH, C64(64)), TL);
          Set(B.CreateBinaryIntrinsic(Intrinsic::umin, R, C64(W)), C64(0));
          break;
        }
        case Intrinsic::ctpop:
          Set(B.CreateAdd(B.CreateUnaryIntrinsic(Intrinsic::ctpop, L1),
                          B.CreateUnaryIntrinsic(Intrinsic::ctpop, H1)),
              C64(0));
          break;
        case Intrinsic::umin:
        case Intrinsic::umax:
        case Intrinsic::smin:
        case Intrinsic::smax: {
          Limbs2 Y = limbsOf(II->getArgOperand(1));
          CmpInst::Predicate P =
              StringSwitch<CmpInst::Predicate>(
                  Intrinsic::getBaseName(II->getIntrinsicID()))
                  .Case("llvm.umin", CmpInst::ICMP_ULT)
                  .Case("llvm.umax", CmpInst::ICMP_UGT)
                  .Case("llvm.smin", CmpInst::ICMP_SLT)
                  .Default(CmpInst::ICMP_SGT);
          Value *X = icmp(B, P, {L1, H1}, Y, W);
          Limbs[I] = {B.CreateSelect(X, L1, Y.first),
                      B.CreateSelect(X, H1, Y.second)};
          break;
        }
        default:
          report_fatal_error(Twine("AIRWriter: unhandled wide integer "
                                   "intrinsic '") +
                             II->getCalledFunction()->getName() + "'");
        }
      } else {
        report_fatal_error(Twine("AIRWriter: unhandled wide integer user '") +
                           I->getOpcodeName() + "'" +
                           (isa<LoadInst>(I) || isa<StoreInst>(I) ||
                                    isa<AtomicRMWInst>(I) ||
                                    isa<AtomicCmpXchgInst>(I)
                                ? " (AIR has no atomics of this width)"
                                : ""));
      }
    }
    for (auto &[Phi, LH] : Phis)
      for (unsigned K = 0; K < Phi->getNumIncomingValues(); ++K) {
        auto [L, H] = limbsOf(Phi->getIncomingValue(K));
        LH.first->addIncoming(L, Phi->getIncomingBlock(K));
        LH.second->addIncoming(H, Phi->getIncomingBlock(K));
      }
    for (Instruction *I : Wide)
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    for (auto It = Wide.rbegin(); It != Wide.rend(); ++It)
      (*It)->eraseFromParent();
  }
  // Declarations of wide intrinsics left unused.
  for (Function &D : make_early_inc_range(M))
    if (D.isIntrinsic() && D.use_empty() &&
        (isWide(D.getReturnType()) ||
         any_of(D.getFunctionType()->params(), isWide)))
      D.eraseFromParent();
}

void stripLifetimeIntrinsics(Module &M) {
  SmallVector<Instruction *, 16> Dead;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction()) {
            Intrinsic::ID ID = Callee->getIntrinsicID();
            if (ID == Intrinsic::lifetime_start ||
                ID == Intrinsic::lifetime_end)
              Dead.push_back(CI);
          }
  for (auto *I : Dead)
    I->eraseFromParent();

  SmallVector<Function *, 4> DeadDecls;
  for (auto &F : M) {
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end) &&
        F.use_empty())
      DeadDecls.push_back(&F);
  }
  for (auto *F : DeadDecls)
    F->eraseFromParent();
}

void canonicalizeNNegZExt(Module &M) {
  SmallVector<ZExtInst *, 16> Zexts;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *ZE = dyn_cast<ZExtInst>(&I))
          if (ZE->hasNonNeg())
            Zexts.push_back(ZE);
  for (auto *ZE : Zexts) {
    auto *SE = CastInst::Create(Instruction::SExt, ZE->getOperand(0),
                                ZE->getType(), "", ZE->getIterator());
    SE->takeName(ZE);
    ZE->replaceAllUsesWith(SE);
    ZE->eraseFromParent();
  }
}

void lowerFreezeInsts(Module &M) {
  SmallVector<FreezeInst *, 8> Frozen;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *FI = dyn_cast<FreezeInst>(&I))
          Frozen.push_back(FI);
  for (auto *FI : Frozen) {
    FI->replaceAllUsesWith(FI->getOperand(0));
    FI->eraseFromParent();
  }
}

void stripDisjointFlags(Module &M) {
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *BO = dyn_cast<PossiblyDisjointInst>(&I))
          if (BO->isDisjoint())
            BO->setIsDisjoint(false);
}

// Generic pointers (agx-raise-to-air's address-space hint, llvm/IR/AIR.h),
// emulated: Metal has no generic address space. A generic address is an
// integer in ZLUDA's AGX encoding, which its isspacep already tests as bits:
// threadgroup addresses with bit 63 set, private (thread) addresses with bits
// 63-62 = 01, device and constant addresses as they are (below 2^62; Metal's
// threadgroup and private addresses are small integers, an M2 Max's device
// addresses about 2^36). Casts to and from generic add and strip the tag; an
// access through a generic pointer goes to the address space its origin shows
// (a cast from it, through GEPs, phis and selects) or, unknown, to the one its
// tag names, tested at run time. When the module casts no threadgroup
// (private) address to generic, no generic address is one, and that branch
// is left out: code that never mixes address spaces pays nothing.
namespace generic {
constexpr uint64_t SharedTag = 1ull << 63, PrivateTag = 1ull << 62;
enum Space { Device, Shared, Private, Unknown };

Space spaceOf(Value *P, SmallPtrSetImpl<Value *> &Seen) {
  if (!Seen.insert(P).second)
    return Device; // a cycle adds no origin
  if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(P)) {
    switch (ASC->getSrcAddressSpace()) {
    case 3:
      return Shared;
    case 0:
      return Private;
    case 1:
    case 2:
      return Device;
    default:
      return Unknown;
    }
  }
  if (auto *GEP = dyn_cast<GEPOperator>(P))
    return spaceOf(GEP->getPointerOperand(), Seen);
  auto Merge = [&](ArrayRef<Value *> Vs) {
    std::optional<Space> R;
    for (Value *V : Vs) {
      Space S = spaceOf(V, Seen);
      if (R && *R != S)
        return Unknown;
      R = S;
    }
    return R.value_or(Unknown);
  };
  if (auto *PN = dyn_cast<PHINode>(P))
    return Merge(SmallVector<Value *, 4>(PN->incoming_values()));
  if (auto *SI = dyn_cast<SelectInst>(P))
    return Merge({SI->getTrueValue(), SI->getFalseValue()});
  if (isa<ConstantPointerNull>(P))
    return Device;
  return Unknown;
}

Value *pointerOperand(Instruction *I) {
  if (auto *L = dyn_cast<LoadInst>(I))
    return L->getPointerOperand();
  if (auto *S = dyn_cast<StoreInst>(I))
    return S->getPointerOperand();
  if (auto *R = dyn_cast<AtomicRMWInst>(I))
    return R->getPointerOperand();
  if (auto *C = dyn_cast<AtomicCmpXchgInst>(I))
    return C->getPointerOperand();
  return nullptr;
}

// P, a generic pointer, as a pointer into address space AS (its tag cleared).
Value *untag(IRBuilder<> &B, Value *P, unsigned AS) {
  auto *PT = PointerType::get(B.getContext(), AS);
  if (AS == 1)
    return B.CreateAddrSpaceCast(P, PT);
  Value *I = B.CreatePtrToInt(P, B.getInt64Ty());
  I = B.CreateAnd(I, ~(AS == 3 ? SharedTag : PrivateTag));
  return B.CreateIntToPtr(I, PT);
}

// I, a memory access, through Ptr instead.
void retarget(Instruction *I, Value *Ptr) {
  if (auto *S = dyn_cast<StoreInst>(I))
    S->setOperand(1, Ptr);
  else
    I->setOperand(0, Ptr);
}

void emulate(Module &M) {
  bool AnyShared = false, AnyPrivate = false;
  SmallVector<AddrSpaceCastInst *, 8> Casts;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
        unsigned From = ASC->getSrcAddressSpace(), To = ASC->getDestAddressSpace();
        if (To == air::GenericHintAS) {
          AnyShared |= From == 3;
          AnyPrivate |= From == 0;
        }
        if (To == air::GenericHintAS || From == air::GenericHintAS)
          Casts.push_back(ASC);
      }

  // The accesses, classified while the casts still show the origins.
  SmallVector<std::pair<Instruction *, Space>, 16> Accesses;
  SmallVector<MemIntrinsic *, 4> MemOps;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
        if (any_of(MI->args(), [](Value *A) {
              return A->getType()->isPointerTy() &&
                     A->getType()->getPointerAddressSpace() == air::GenericHintAS;
            }))
          MemOps.push_back(MI);
        continue;
      }
      Value *P = pointerOperand(&I);
      if (!P || P->getType()->getPointerAddressSpace() != air::GenericHintAS)
        continue;
      SmallPtrSet<Value *, 8> Seen;
      Space S = spaceOf(P, Seen);
      if (S == Unknown && !AnyShared && !AnyPrivate)
        S = Device;
      Accesses.push_back({&I, S});
    }
  // Generic memcpy / memset / memmove: as loops of loads and stores, which
  // are then dispatched like the others.
  if (!MemOps.empty()) {
    TargetTransformInfo TTI(M.getDataLayout());
    for (MemIntrinsic *MI : MemOps) {
      if (auto *MC = dyn_cast<MemCpyInst>(MI))
        expandMemCpyAsLoop(MC, TTI);
      else if (auto *MS = dyn_cast<MemSetInst>(MI))
        expandMemSetAsLoop(MS);
      else if (auto *MM = dyn_cast<MemMoveInst>(MI))
        expandMemMoveAsLoop(MM, TTI);
      else
        continue;
      MI->eraseFromParent();
    }
    // the loops' accesses, through the same pointers
    if (!MemOps.empty()) {
      Accesses.clear();
      for (Function &F : M)
        for (Instruction &I : instructions(F)) {
          Value *P = pointerOperand(&I);
          if (!P || P->getType()->getPointerAddressSpace() != air::GenericHintAS)
            continue;
          SmallPtrSet<Value *, 8> Seen;
          Space S = spaceOf(P, Seen);
          if (S == Unknown && !AnyShared && !AnyPrivate)
            S = Device;
          Accesses.push_back({&I, S});
        }
    }
  }

  for (auto [I, S] : Accesses) {
    Value *P = pointerOperand(I);
    IRBuilder<> B(I);
    if (S != Unknown) {
      retarget(I, untag(B, P, S == Device ? 1 : S == Shared ? 3 : 0));
      continue;
    }
    // Unknown origin: by the tag, in a block per address space.
    Value *Bits = B.CreatePtrToInt(P, B.getInt64Ty());
    BasicBlock *Head = I->getParent();
    BasicBlock *Tail = SplitBlock(Head, I);
    Function *F = Head->getParent();
    SmallVector<std::pair<unsigned, BasicBlock *>, 3> Arms;
    Arms.push_back({1, BasicBlock::Create(M.getContext(), "generic.device", F, Tail)});
    if (AnyShared)
      Arms.push_back({3, BasicBlock::Create(M.getContext(), "generic.shared", F, Tail)});
    if (AnyPrivate)
      Arms.push_back({0, BasicBlock::Create(M.getContext(), "generic.private", F, Tail)});
    Head->getTerminator()->eraseFromParent();
    B.SetInsertPoint(Head);
    if (Arms.size() == 2) {
      Value *IsOther = Arms[1].first == 3
                           ? B.CreateICmpSLT(Bits, B.getInt64(0))
                           : B.CreateICmpEQ(B.CreateLShr(Bits, 62), B.getInt64(1));
      B.CreateCondBr(IsOther, Arms[1].second, Arms[0].second);
    } else {
      // (no switch: the AIR writer has none, LowerSwitch has run)
      BasicBlock *NotShared =
          BasicBlock::Create(M.getContext(), "generic.notshared", F, Arms[0].second);
      B.CreateCondBr(B.CreateICmpSLT(Bits, B.getInt64(0)), Arms[1].second, NotShared);
      IRBuilder<> NB(NotShared);
      NB.CreateCondBr(NB.CreateICmpEQ(NB.CreateLShr(Bits, 62), NB.getInt64(1)),
                      Arms[2].second, Arms[0].second);
    }
    PHINode *Phi = nullptr;
    if (!I->getType()->isVoidTy()) {
      Phi = PHINode::Create(I->getType(), Arms.size(), "", Tail->begin());
      I->replaceAllUsesWith(Phi);
    }
    for (auto [AS, BB] : Arms) {
      IRBuilder<> AB(BB);
      Value *Ptr = untag(AB, P, AS);
      Value *Result = nullptr;
      if (AS == 0 && isa<AtomicRMWInst, AtomicCmpXchgInst>(I)) {
        // private memory belongs to one thread: its atomics are plain
        // accesses
        if (auto *R = dyn_cast<AtomicRMWInst>(I)) {
          LoadInst *Old = AB.CreateAlignedLoad(R->getType(), Ptr, R->getAlign(), R->isVolatile());
          Value *New = buildAtomicRMWValue(R->getOperation(), AB, Old, R->getValOperand());
          AB.CreateAlignedStore(New, Ptr, R->getAlign(), R->isVolatile());
          Result = Old;
        } else {
          auto *X = cast<AtomicCmpXchgInst>(I);
          Type *T = X->getNewValOperand()->getType();
          LoadInst *Old = AB.CreateAlignedLoad(T, Ptr, X->getAlign(), X->isVolatile());
          Value *Eq = AB.CreateICmpEQ(Old, X->getCompareOperand());
          AB.CreateAlignedStore(AB.CreateSelect(Eq, X->getNewValOperand(), Old), Ptr,
                                X->getAlign(), X->isVolatile());
          Result = AB.CreateInsertValue(
              AB.CreateInsertValue(PoisonValue::get(X->getType()), Old, 0), Eq, 1);
        }
      } else {
        Instruction *C = I->clone();
        AB.Insert(C);
        retarget(C, Ptr);
        if (AS == 0) {
          if (auto *L = dyn_cast<LoadInst>(C))
            L->setAtomic(AtomicOrdering::NotAtomic);
          else if (auto *St = dyn_cast<StoreInst>(C))
            St->setAtomic(AtomicOrdering::NotAtomic);
        }
        Result = C;
      }
      AB.CreateBr(Tail);
      if (Phi)
        Phi->addIncoming(Result, BB);
    }
    I->eraseFromParent();
  }

  // Casts: into generic adds the tag, out of it strips it.
  for (AddrSpaceCastInst *ASC : Casts) {
    unsigned From = ASC->getSrcAddressSpace(), To = ASC->getDestAddressSpace();
    IRBuilder<> B(ASC);
    Value *R = nullptr;
    if (To == air::GenericHintAS && (From == 3 || From == 0)) {
      Value *I = B.CreatePtrToInt(ASC->getOperand(0), B.getInt64Ty());
      R = B.CreateIntToPtr(B.CreateOr(I, From == 3 ? SharedTag : PrivateTag),
                           ASC->getType());
    } else if (From == air::GenericHintAS && (To == 3 || To == 0)) {
      R = untag(B, ASC->getOperand(0), To);
    }
    if (R) {
      R->takeName(ASC);
      ASC->replaceAllUsesWith(R);
      ASC->eraseFromParent();
    }
  }
}
} // namespace generic

// Generic pointers (agx-raise-to-air's address-space hint) as device
// pointers, after generic::emulate has made their accesses and casts
// explicit.
void genericToDevice(Module &M) {
  LLVMContext &Ctx = M.getContext();
  auto *Generic = PointerType::get(Ctx, air::GenericHintAS);
  auto *Device = PointerType::get(Ctx, 1);
  // Constant expressions on threadgroup variables (a generic address of one,
  // PTX's cvta.shared, under a ptrtoint whose type maps to nothing) as
  // instructions, which the mapping below reaches: left as constants, a
  // generic cast inside them crashed Metal's compiler.
  SmallVector<Constant *, 8> TGGlobals;
  for (GlobalVariable &GV : M.globals())
    if (GV.getAddressSpace() == 3)
      TGGlobals.push_back(&GV);
  if (!TGGlobals.empty())
    convertUsersOfConstantsToInstructions(TGGlobals);
  generic::emulate(M);
  auto Map = [&](Type *T) -> Type * {
    if (T == Generic)
      return Device;
    if (auto *VT = dyn_cast<VectorType>(T); VT && VT->getElementType() == Generic)
      return VectorType::get(Device, VT->getElementCount());
    return nullptr;
  };
  auto MapConstant = [&](Constant *C) -> Constant * {
    Type *T = Map(C->getType());
    if (!T)
      return nullptr;
    if (isa<ConstantPointerNull>(C))
      return ConstantPointerNull::get(cast<PointerType>(T));
    if (isa<PoisonValue>(C))
      return PoisonValue::get(T);
    if (isa<UndefValue>(C))
      return UndefValue::get(T);
    if (auto *CE = dyn_cast<ConstantExpr>(C);
        CE && CE->getOpcode() == Instruction::IntToPtr)
      return ConstantExpr::getIntToPtr(CE->getOperand(0), T);
    return ConstantExpr::getAddrSpaceCast(C, T);
  };
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (Map(F.getReturnType()) ||
        any_of(F.getFunctionType()->params(), Map)) {
      F.getContext().diagnose(DiagnosticInfoUnsupported(
          F, "a generic pointer in the signature of a function"));
      continue;
    }
    SmallVector<CallBase *, 4> Intrinsics;
    for (Instruction &I : instructions(F)) {
      for (Use &U : I.operands())
        if (auto *C = dyn_cast<Constant>(U.get()); C && !isa<GlobalValue>(C))
          if (Constant *NC = MapConstant(C))
            U.set(NC);
      if (Type *T = Map(I.getType()))
        I.mutateType(T);
      if (auto *CB = dyn_cast<CallBase>(&I);
          CB && CB->getCalledFunction() && CB->getCalledFunction()->isIntrinsic())
        Intrinsics.push_back(CB);
    }
    // Intrinsics overloaded on the pointer type (memcpy ...) take the device
    // overload.
    for (CallBase *CB : Intrinsics) {
      Function *Callee = CB->getCalledFunction();
      SmallVector<Type *, 4> ArgTys;
      for (Value *A : CB->args())
        ArgTys.push_back(A->getType());
      auto *FT = FunctionType::get(CB->getType(), ArgTys, false);
      if (FT == Callee->getFunctionType())
        continue;
      SmallVector<Type *, 2> Overloads;
      if (!Intrinsic::isSignatureValid(Callee->getIntrinsicID(), FT, Overloads))
        continue;
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          &M, Callee->getIntrinsicID(), Overloads);
      CB->mutateFunctionType(Decl->getFunctionType());
      CB->setCalledFunction(Decl);
    }
    // Casts between generic and device pointers are now no-ops.
    for (Instruction &I : make_early_inc_range(instructions(F)))
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I);
          ASC && ASC->getSrcTy() == ASC->getDestTy()) {
        ASC->replaceAllUsesWith(ASC->getOperand(0));
        ASC->eraseFromParent();
      }
  }
}

// The overload suffix of AIR's math functions for a float type (f16, f32,
// v4f32), or "" for other types.
std::string airFloatSuffix(Type *T) {
  std::string S;
  if (auto *VT = dyn_cast<FixedVectorType>(T)) {
    if (VT->getNumElements() > 4)
      return "";
    S = "v" + utostr(VT->getNumElements());
    T = VT->getElementType();
  }
  if (T->isHalfTy())
    return S + "f16";
  if (T->isFloatTy())
    return S + "f32";
  return "";
}

CallInst *callAIRMath(IRBuilder<> &B, StringRef Name, Type *RetTy,
                      ArrayRef<Value *> Args, bool ReadNone = true) {
  SmallVector<Type *, 2> Tys;
  for (Value *A : Args)
    Tys.push_back(A->getType());
  Module &M = *B.GetInsertBlock()->getModule();
  FunctionCallee F =
      M.getOrInsertFunction(Name, FunctionType::get(RetTy, Tys, false));
  if (auto *Fn = dyn_cast<Function>(F.getCallee())) {
    Fn->setDoesNotThrow();
    Fn->setWillReturn();
    if (ReadNone)
      Fn->setDoesNotAccessMemory();
  }
  return B.CreateCall(F, Args);
}

// Replaces CI, which returns a two-element struct, with Fields: its
// extractvalue users directly, any other user through a rebuilt struct.
void replaceStructResult(CallInst *CI, Value *F0, Value *F1) {
  Value *Fields[] = {F0, F1};
  for (User *U : make_early_inc_range(CI->users()))
    if (auto *EV = dyn_cast<ExtractValueInst>(U);
        EV && EV->getNumIndices() == 1) {
      EV->replaceAllUsesWith(Fields[EV->getIndices()[0]]);
      EV->eraseFromParent();
    }
  if (!CI->use_empty()) {
    IRBuilder<> B(CI);
    Value *S = B.CreateInsertValue(PoisonValue::get(CI->getType()), F0, 0);
    CI->replaceAllUsesWith(B.CreateInsertValue(S, F1, 1));
  }
  CI->eraseFromParent();
}

// llvm.is.fpclass as integer tests of the encoding: AIR has no function for
// it.
Value *expandIsFPClass(IRBuilder<> &B, Value *X, FPClassTest Mask) {
  Type *FTy = X->getType();
  const fltSemantics &Sem = FTy->getScalarType()->getFltSemantics();
  unsigned W = FTy->getScalarSizeInBits();
  Type *ITy = FTy->getWithNewType(B.getIntNTy(W));
  auto C = [&](const APInt &V) { return ConstantInt::get(ITy, V); };
  APInt Inf = APFloat::getInf(Sem).bitcastToAPInt();
  APInt QNaN = APFloat::getQNaN(Sem).bitcastToAPInt();
  APInt MinNorm = APFloat::getSmallestNormalized(Sem).bitcastToAPInt();
  Value *Bits = B.CreateBitCast(X, ITy);
  Value *Abs = B.CreateAnd(Bits, C(APInt::getSignedMaxValue(W)));
  Value *R = nullptr;
  auto Or = [&](Value *T) { R = R ? B.CreateOr(R, T) : T; };
  if ((Mask & fcNan) == fcNan)
    Or(B.CreateICmpUGT(Abs, C(Inf)));
  else if (Mask & fcSNan)
    Or(B.CreateAnd(B.CreateICmpUGT(Abs, C(Inf)), B.CreateICmpULT(Abs, C(QNaN))));
  else if (Mask & fcQNan)
    Or(B.CreateICmpUGE(Abs, C(QNaN)));
  // The other classes come in pairs, by sign.
  auto Signed = [&](FPClassTest NegC, FPClassTest PosC, Value *T) {
    if ((Mask & NegC) && (Mask & PosC))
      Or(T);
    else
      Or(B.CreateAnd(T, B.CreateICmp(Mask & NegC ? ICmpInst::ICMP_SLT
                                                  : ICmpInst::ICMP_SGE,
                                     Bits, Constant::getNullValue(ITy))));
  };
  if (Mask & fcInf)
    Signed(fcNegInf, fcPosInf, B.CreateICmpEQ(Abs, C(Inf)));
  if (Mask & fcNormal)
    Signed(fcNegNormal, fcPosNormal,
           B.CreateICmpULT(B.CreateSub(Abs, C(MinNorm)), C(Inf - MinNorm)));
  if (Mask & fcSubnormal)
    Signed(fcNegSubnormal, fcPosSubnormal,
           B.CreateICmpULT(B.CreateSub(Abs, C(APInt(W, 1))),
                           C(MinNorm - 1)));
  if (Mask & fcZero)
    Signed(fcNegZero, fcPosZero,
           B.CreateICmpEQ(Abs, Constant::getNullValue(ITy)));
  return R ? R : Constant::getNullValue(FTy->getWithNewType(B.getInt1Ty()));
}

// What Metal miscompiles, crashes on or leaves undefined, found on the M2 Max
// (llvm/utils/air-gpu, groups llvm-float-ieee, llvm-int-ops, llvm-conv and
// llvm-memory-ir), in terms of what it compiles correctly:
//
// * frem: Metal's is inexact (up to 1e9 ulps), gives NaN for x frem inf and
//   +0 for -3 frem 1. air.fmod.f32 is exact, with C's special cases.
// * llvm.round: air.round(-0) is +0, so the sign is copied back.
// * llvm.sadd/ssub.with.overflow: Metal sets the overflow bit wrongly for
//   about half of the operands.
// * llvm.fptosi.sat, llvm.fptoui.sat: crash Metal's compiler.
// * llvm.ldexp, llvm.frexp, llvm.is.fpclass: undefined symbols. AIR has
//   air.ldexp(x, i32) and air.frexp(x, thread i32 *), whose mantissa for -0
//   is +0.
// * llvm.roundeven, llvm.nearbyint: crash Metal's compiler; they are
//   llvm.rint (air.rint) in the fixed rounding mode.
// * llvm.powi, llvm.lround, llvm.llround, llvm.lrint, llvm.llrint,
//   llvm.canonicalize, llvm.smul/umul.with.overflow wider than 32 bits: crash
//   Metal's compiler. llvm.minimumnum, llvm.maximumnum, llvm.sincos,
//   llvm.smulh, llvm.umulh: undefined symbols.
// * llvm.fabs of half or bfloat: air.fabs.f16 canonicalizes a NaN (0x7E00);
//   the sign bit is cleared in the bits.
void expandIntrinsicsMetalMishandles(Module &M) {
  static constexpr Intrinsic::ID Expanded[] = {
      Intrinsic::round,       Intrinsic::roundeven,  Intrinsic::nearbyint,
      Intrinsic::sadd_with_overflow, Intrinsic::ssub_with_overflow,
      Intrinsic::smul_with_overflow, Intrinsic::umul_with_overflow,
      Intrinsic::fptosi_sat,  Intrinsic::fptoui_sat, Intrinsic::is_fpclass,
      Intrinsic::ldexp,       Intrinsic::frexp,      Intrinsic::powi,
      Intrinsic::lround,      Intrinsic::llround,    Intrinsic::lrint,
      Intrinsic::llrint,      Intrinsic::canonicalize,
      Intrinsic::minimumnum,  Intrinsic::maximumnum, Intrinsic::sincos,
      Intrinsic::fabs,        Intrinsic::smulh,      Intrinsic::umulh};
  SmallVector<Instruction *, 16> Work;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      if (I.getOpcode() == Instruction::FRem) {
        if (!airFloatSuffix(I.getType()).empty())
          Work.push_back(&I);
        continue;
      }
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II || !is_contained(Expanded, II->getIntrinsicID()))
        continue;
      // AIR's ldexp and frexp exist for half and float, up to 4 lanes.
      if ((II->getIntrinsicID() == Intrinsic::ldexp ||
           II->getIntrinsicID() == Intrinsic::frexp) &&
          airFloatSuffix(II->getArgOperand(0)->getType()).empty())
        continue;
      // Metal multiplies 32-bit (and narrower) operands with overflow right.
      if ((II->getIntrinsicID() == Intrinsic::smul_with_overflow ||
           II->getIntrinsicID() == Intrinsic::umul_with_overflow) &&
          II->getArgOperand(0)->getType()->getScalarSizeInBits() <= 32)
        continue;
      // Only half and bfloat lose NaN payloads in air.fabs.
      if (II->getIntrinsicID() == Intrinsic::fabs &&
          II->getType()->getScalarSizeInBits() != 16)
        continue;
      Work.push_back(II);
    }

  for (Instruction *I : Work) {
    IRBuilder<> B(I);
    if (I->getOpcode() == Instruction::FRem) {
      // air.fmod.f16 has x fmod inf = NaN; in float, the remainder is exact
      // and so is its truncation back to half.
      Type *Ty = I->getType();
      Type *FTy = Ty->getWithNewType(B.getFloatTy());
      Value *X = B.CreateFPExt(I->getOperand(0), FTy);
      Value *Y = B.CreateFPExt(I->getOperand(1), FTy);
      Value *R = B.CreateFPTrunc(
          callAIRMath(B, "air.fmod." + airFloatSuffix(FTy), FTy, {X, Y}), Ty);
      R->takeName(I);
      I->replaceAllUsesWith(R);
      I->eraseFromParent();
      continue;
    }
    auto *CI = cast<IntrinsicInst>(I);
    Value *X = CI->getArgOperand(0);
    switch (CI->getIntrinsicID()) {
    case Intrinsic::round: {
      B.SetInsertPoint(CI->getNextNode());
      B.setFastMathFlags(CI->getFastMathFlags());
      Value *R = B.CreateCopySign(CI, X);
      CI->replaceUsesWithIf(R, [&](Use &U) { return U.getUser() != R; });
      break;
    }
    case Intrinsic::roundeven:
    case Intrinsic::nearbyint: {
      Value *R = B.CreateUnaryIntrinsic(Intrinsic::rint, X, CI);
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::ssub_with_overflow: {
      Value *Y = CI->getArgOperand(1);
      bool Add = CI->getIntrinsicID() == Intrinsic::sadd_with_overflow;
      Value *R = Add ? B.CreateAdd(X, Y) : B.CreateSub(X, Y);
      // Overflow: the result's sign differs from x's and from that of y
      // (add) or of -y (sub).
      Value *S = B.CreateAnd(B.CreateXor(X, R),
                             Add ? B.CreateXor(Y, R) : B.CreateXor(X, Y));
      replaceStructResult(
          CI, R, B.CreateICmpSLT(S, Constant::getNullValue(S->getType())));
      break;
    }
    case Intrinsic::fptosi_sat:
    case Intrinsic::fptoui_sat: {
      // The bounds rounded toward zero: beyond them the result saturates.
      bool Signed = CI->getIntrinsicID() == Intrinsic::fptosi_sat;
      Type *ITy = CI->getType();
      unsigned W = ITy->getScalarSizeInBits();
      const fltSemantics &Sem = X->getType()->getScalarType()->getFltSemantics();
      APInt Min = Signed ? APInt::getSignedMinValue(W) : APInt::getMinValue(W);
      APInt Max = Signed ? APInt::getSignedMaxValue(W) : APInt::getMaxValue(W);
      APFloat MinF(Sem), MaxF(Sem);
      MinF.convertFromAPInt(Min, Signed, APFloat::rmTowardZero);
      MaxF.convertFromAPInt(Max, Signed, APFloat::rmTowardZero);
      // A vector splat as ConstantDataVector (not a vector ConstantFP).
      auto FC = [&](const APFloat &V) -> Constant * {
        Constant *C = ConstantFP::get(M.getContext(), V);
        if (auto *VT = dyn_cast<VectorType>(X->getType()))
          return ConstantVector::getSplat(VT->getElementCount(), C);
        return C;
      };
      Value *R = Signed ? B.CreateFPToSI(X, ITy) : B.CreateFPToUI(X, ITy);
      R = B.CreateSelect(B.CreateFCmpULT(X, FC(MinF)), ConstantInt::get(ITy, Min),
                         R);
      R = B.CreateSelect(B.CreateFCmpOGT(X, FC(MaxF)), ConstantInt::get(ITy, Max),
                         R);
      R = B.CreateSelect(B.CreateFCmpUNO(X, X), Constant::getNullValue(ITy), R);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::is_fpclass: {
      auto Mask = static_cast<FPClassTest>(
          cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue());
      CI->replaceAllUsesWith(expandIsFPClass(B, X, Mask));
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::ldexp: {
      // AIR's exponent is an i32: a wider one is clamped, which does not
      // change the result.
      Value *E = CI->getArgOperand(1);
      Type *I32Ty = E->getType()->getWithNewBitWidth(32);
      unsigned EW = E->getType()->getScalarSizeInBits();
      if (EW > 32) {
        E = B.CreateBinaryIntrinsic(
            Intrinsic::smin, E,
            ConstantInt::get(E->getType(), APInt::getSignedMaxValue(32).sext(EW)));
        E = B.CreateBinaryIntrinsic(
            Intrinsic::smax, E,
            ConstantInt::get(E->getType(), APInt::getSignedMinValue(32).sext(EW)));
      }
      E = B.CreateSExtOrTrunc(E, I32Ty);
      CallInst *R = callAIRMath(B, "air.ldexp." + airFloatSuffix(X->getType()),
                                X->getType(), {X, E});
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::fabs: {
      Type *ITy = X->getType()->getWithNewType(B.getInt16Ty());
      Value *R = B.CreateBitCast(
          B.CreateAnd(B.CreateBitCast(X, ITy), ConstantInt::get(ITy, 0x7fff)),
          X->getType());
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::canonicalize: {
      // The hardware flushes f32 subnormals where they are used, and NaNs
      // are quiet: the value itself is canonical enough.
      CI->replaceAllUsesWith(X);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::powi: {
      // |x|^n by pow (Metal's pow of a negative base is undefined), negated
      // for a negative base and an odd exponent.
      Value *N = CI->getArgOperand(1);
      Type *FTy = X->getType();
      Value *NF = B.CreateSIToFP(N, FTy);
      Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
      Value *P = B.CreateBinaryIntrinsic(Intrinsic::pow, Abs, NF, CI);
      Value *Odd = B.CreateTrunc(N, N->getType()->getWithNewBitWidth(1));
      Value *Neg =
          B.CreateAnd(Odd, B.CreateFCmpOLT(X, ConstantFP::get(FTy, 0.0)));
      // (fsub, not fneg: AIRLowerFNeg has run)
      Value *R = B.CreateSelect(
          Neg, B.CreateFSub(ConstantFP::getNegativeZero(FTy), P), P);
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::lround:
    case Intrinsic::llround:
    case Intrinsic::lrint:
    case Intrinsic::llrint: {
      bool Round = CI->getIntrinsicID() == Intrinsic::lround ||
                   CI->getIntrinsicID() == Intrinsic::llround;
      // The integer of the rounded value; out of range it is unspecified.
      Value *Rounded = B.CreateUnaryIntrinsic(
          Round ? Intrinsic::round : Intrinsic::rint, X);
      Value *R = B.CreateFPToSI(Rounded, CI->getType());
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum: {
      // minnum / maxnum (a NaN operand is missing), with -0 below +0: for
      // equal operands, OR (minimum) or AND (maximum) of their encodings.
      bool Min = CI->getIntrinsicID() == Intrinsic::minimumnum;
      Value *Y = CI->getArgOperand(1);
      Type *FTy = X->getType();
      Type *ITy = FTy->getWithNewType(B.getIntNTy(FTy->getScalarSizeInBits()));
      Value *MM = B.CreateBinaryIntrinsic(
          Min ? Intrinsic::minnum : Intrinsic::maxnum, X, Y, CI);
      Value *XB = B.CreateBitCast(X, ITy), *YB = B.CreateBitCast(Y, ITy);
      Value *Eq =
          B.CreateBitCast(Min ? B.CreateOr(XB, YB) : B.CreateAnd(XB, YB), FTy);
      Value *R = B.CreateSelect(B.CreateFCmpOEQ(X, Y), Eq, MM);
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::smulh:
    case Intrinsic::umulh: {
      // The high half of the double-width product (a wide product of i64
      // operands is split into limbs by expandWideIntegers).
      bool Signed = CI->getIntrinsicID() == Intrinsic::smulh;
      Type *ITy = X->getType();
      unsigned W = ITy->getScalarSizeInBits();
      Type *WTy = ITy->getWithNewBitWidth(2 * W);
      auto Ext = [&](Value *V) {
        return Signed ? B.CreateSExt(V, WTy) : B.CreateZExt(V, WTy);
      };
      Value *P = B.CreateMul(Ext(X), Ext(CI->getArgOperand(1)));
      Value *R = B.CreateTrunc(B.CreateLShr(P, W), ITy);
      R->takeName(CI);
      CI->replaceAllUsesWith(R);
      CI->eraseFromParent();
      break;
    }
    case Intrinsic::sincos: {
      Value *Sin = B.CreateUnaryIntrinsic(Intrinsic::sin, X, CI);
      Value *Cos = B.CreateUnaryIntrinsic(Intrinsic::cos, X, CI);
      replaceStructResult(CI, Sin, Cos);
      break;
    }
    case Intrinsic::smul_with_overflow:
    case Intrinsic::umul_with_overflow: {
      // The product overflows iff dividing it by a nonzero x does not give
      // y back, or (signed) it is -1 * INT_MIN, which the division misses.
      Value *Y = CI->getArgOperand(1);
      Type *ITy = X->getType();
      bool Signed = CI->getIntrinsicID() == Intrinsic::smul_with_overflow;
      Value *R = B.CreateMul(X, Y);
      Value *XNZ = B.CreateICmpNE(X, Constant::getNullValue(ITy));
      // a divisor of 1 where x is 0, so that the division is defined
      Value *D = B.CreateSelect(XNZ, X, ConstantInt::get(ITy, 1));
      Value *Q = Signed ? B.CreateSDiv(R, D) : B.CreateUDiv(R, D);
      Value *Ovf = B.CreateAnd(XNZ, B.CreateICmpNE(Q, Y));
      if (Signed) {
        unsigned W = ITy->getScalarSizeInBits();
        Value *MinInt = ConstantInt::get(ITy, APInt::getSignedMinValue(W));
        Value *AllOnes = Constant::getAllOnesValue(ITy);
        Ovf = B.CreateOr(
            Ovf, B.CreateOr(B.CreateAnd(B.CreateICmpEQ(X, AllOnes),
                                        B.CreateICmpEQ(Y, MinInt)),
                            B.CreateAnd(B.CreateICmpEQ(Y, AllOnes),
                                        B.CreateICmpEQ(X, MinInt))));
      }
      replaceStructResult(CI, R, Ovf);
      break;
    }
    case Intrinsic::frexp: {
      auto *STy = cast<StructType>(CI->getType());
      Type *ETy = STy->getElementType(1);
      Type *I32Ty = ETy->getWithNewBitWidth(32);
      IRBuilder<> EB(&*CI->getFunction()->getEntryBlock().getFirstInsertionPt());
      AllocaInst *Slot = EB.CreateAlloca(I32Ty);
      CallInst *Mant = callAIRMath(B, "air.frexp." + airFloatSuffix(X->getType()),
                                   X->getType(), {X, Slot}, false);
      Value *E = B.CreateSExtOrTrunc(B.CreateLoad(I32Ty, Slot), ETy);
      replaceStructResult(CI, B.CreateCopySign(Mant, X), E);
      break;
    }
    default:
      llvm_unreachable("unexpected intrinsic");
    }
  }

  for (Function &F : make_early_inc_range(M))
    if (F.use_empty() && is_contained(Expanded, F.getIntrinsicID()))
      F.eraseFromParent();
}

// Threadgroup globals declared, not defined (agx-raise-to-air keeps PTX's
// .shared variables so; AGX lays them out itself). Metal resolves no
// threadgroup symbol: a sized one is defined here (undefined contents, as
// threadgroup memory is), and a zero-sized one, CUDA's dynamic shared memory
// (extern __shared__ T s[]), whose size is known only at dispatch, becomes
// Metal's threadgroup argument: a new last parameter of each kernel using it,
// with threadgroup location index 0 (after any threadgroup parameters the
// kernel already has), whose length the host sets per dispatch
// (setThreadgroupMemoryLength:atIndex:, a multiple of 16). As in CUDA, every
// zero-sized declaration a kernel uses is the same memory, the one argument.
void defineThreadgroupGlobals(Module &M) {
  LLVMContext &Ctx = M.getContext();
  SmallVector<GlobalVariable *, 4> Dynamic;
  for (GlobalVariable &GV : M.globals()) {
    if (!GV.isDeclaration() || GV.getAddressSpace() != 3)
      continue;
    if (M.getDataLayout().getTypeAllocSize(GV.getValueType()).isZero()) {
      Dynamic.push_back(&GV);
      continue;
    }
    GV.setInitializer(UndefValue::get(GV.getValueType()));
    GV.setLinkage(GlobalValue::InternalLinkage);
  }
  if (Dynamic.empty())
    return;

  SmallVector<Constant *, 4> Consts(Dynamic.begin(), Dynamic.end());
  convertUsersOfConstantsToInstructions(Consts);
  // The functions using the dynamic memory, each a kernel (non-kernel
  // functions have been inlined).
  SetVector<Function *> Users;
  for (GlobalVariable *GV : Dynamic)
    for (User *U : GV->users())
      if (auto *I = dyn_cast<Instruction>(U))
        Users.insert(I->getFunction());
  NamedMDNode *Kernels = M.getNamedMetadata("air.kernel");
  Type *I32 = Type::getInt32Ty(Ctx);
  auto C32 = [&](unsigned V) {
    return ConstantAsMetadata::get(ConstantInt::get(I32, V));
  };
  for (Function *F : Users) {
    if (!F->hasFnAttribute("air-kernel")) {
      Ctx.diagnose(DiagnosticInfoUnsupported(
          *F, "dynamic threadgroup memory used outside a kernel"));
      continue;
    }
    SmallVector<Type *, 8> Params(F->getFunctionType()->params());
    unsigned TGIndex = count_if(Params, [](Type *T) {
      return T->isPointerTy() && T->getPointerAddressSpace() == 3;
    });
    Params.push_back(PointerType::get(Ctx, 3));
    auto *NewF = Function::Create(
        FunctionType::get(F->getReturnType(), Params, false), F->getLinkage(),
        F->getAddressSpace(), "", &M);
    NewF->copyAttributesFrom(F);
    NewF->copyMetadata(F, 0);
    NewF->splice(NewF->begin(), F);
    for (auto [Old, New] : zip(F->args(), NewF->args())) {
      New.setName(Old.getName());
      Old.replaceAllUsesWith(&New);
    }
    Argument *TG = NewF->getArg(Params.size() - 1);
    TG->setName(Dynamic.size() == 1 ? Dynamic.front()->getName()
                                    : StringRef("dynamic_threadgroup"));
    for (GlobalVariable *GV : Dynamic)
      GV->replaceUsesWithIf(TG, [&](Use &U) {
        auto *I = dyn_cast<Instruction>(U.getUser());
        return I && I->getFunction() == NewF;
      });
    // Keep a pre-baked !air.kernel pointing at the kernel, and describe the
    // argument there as AIRSystemValues does without one: an "air.buffer" in
    // address space 3 (on the M2 Max an "air.threadgroup" entry crashes
    // Metal's compiler).
    ValueAsMetadata::handleRAUW(F, NewF);
    NewF->takeName(F);
    F->eraseFromParent();
    if (!Kernels)
      continue;
    Align A = Dynamic.front()->getAlign().valueOrOne();
    MDNode *Arg = MDNode::get(
        Ctx, {C32(TG->getArgNo()), MDString::get(Ctx, "air.buffer"),
              MDString::get(Ctx, "air.location_index"), C32(TGIndex), C32(1),
              MDString::get(Ctx, "air.read_write"),
              MDString::get(Ctx, "air.address_space"), C32(3),
              MDString::get(Ctx, "air.arg_type_size"), C32(1),
              MDString::get(Ctx, "air.arg_type_align_size"), C32(A.value()),
              MDString::get(Ctx, "air.arg_type_name"),
              MDString::get(Ctx, "uchar"), MDString::get(Ctx, "air.arg_name"),
              MDString::get(Ctx, TG->getName())});
    for (MDNode *K : Kernels->operands()) {
      auto *KF = K->getNumOperands() == 3
                     ? mdconst::dyn_extract_or_null<Function>(K->getOperand(0))
                     : nullptr;
      auto *Args = KF == NewF ? dyn_cast<MDNode>(K->getOperand(2)) : nullptr;
      if (!Args)
        continue;
      SmallVector<Metadata *, 8> Ops(Args->operands());
      Ops.push_back(Arg);
      K->replaceOperandWith(2, MDNode::get(Ctx, Ops));
    }
  }
  for (GlobalVariable *GV : Dynamic)
    if (GV->use_empty())
      GV->eraseFromParent();
}

// A GEP of an address made by inttoptr as integer arithmetic and an inttoptr
// of its own. ZLUDA forms device addresses so (pointer parameters arrive as
// integers), and the AIR writer gives such an address one pointee type from
// its uses: two byte GEPs of it storing a float and an i32 made Metal's reader
// reject the library ("Unexpected bitcode file", PTX-formalization
// half-oob-set and half-cmp-pk2). Each address then has its own pointee.
void gepsOfIntToPtrAsIntegers(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  for (Function &F : M) {
    bool Changed = true;
    while (Changed) {
      Changed = false;
      // (the bases left unused are erased after the walk: one may be the
      // instruction the walk visits next)
      SmallVector<IntToPtrInst *, 8> Bases;
      for (Instruction &I : make_early_inc_range(instructions(F))) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        auto *Base =
            GEP ? dyn_cast<IntToPtrInst>(GEP->getPointerOperand()) : nullptr;
        if (!Base || GEP->getType()->isVectorTy())
          continue;
        IRBuilder<> B(GEP);
        Type *IntTy = DL.getIntPtrType(GEP->getType());
        Value *Addr = B.CreateZExtOrTrunc(Base->getOperand(0), IntTy);
        Value *Off = emitGEPOffset(&B, DL, GEP, /*NoAssumptions=*/true);
        Value *R = B.CreateIntToPtr(
            B.CreateAdd(Addr, B.CreateSExtOrTrunc(Off, IntTy)), GEP->getType());
        R->takeName(GEP);
        GEP->replaceAllUsesWith(R);
        GEP->eraseFromParent();
        if (!is_contained(Bases, Base))
          Bases.push_back(Base);
        Changed = true;
      }
      for (IntToPtrInst *Base : Bases)
        if (Base->use_empty())
          Base->eraseFromParent();
    }
    // The same address used with different pointees (a memset of it and an
    // i32 store to it, Cycles' intersect_volume_stack): an inttoptr of its
    // own for each use other than the first.
    for (Instruction &I : make_early_inc_range(instructions(F))) {
      auto *ITP = dyn_cast<IntToPtrInst>(&I);
      // (at least two uses: drop_begin of an unused one's uses, Cycles'
      // intersect_closest once its callees were no longer inlined, walked
      // past their end)
      if (!ITP || ITP->getType()->isVectorTy() || !ITP->hasNUsesOrMore(2))
        continue;
      SmallVector<Use *, 8> Uses;
      for (Use &U : ITP->uses())
        Uses.push_back(&U);
      for (Use *U : drop_begin(Uses)) {
        auto *User = cast<Instruction>(U->getUser());
        // a phi's incoming value at the end of its block
        Instruction *At = isa<PHINode>(User)
                              ? cast<PHINode>(User)
                                    ->getIncomingBlock(*U)
                                    ->getTerminator()
                              : User;
        auto *Copy = cast<Instruction>(ITP->clone());
        Copy->insertBefore(At->getIterator());
        U->set(Copy);
      }
    }
  }
}


// Half denormals flushed where the function's denormal mode for half asks for
// it (PTX .ftz on f16 / f16x2: ZLUDA gives such functions
// denormal_fpenv(preservesign) for every type). The GPU keeps half denormals,
// so the arithmetic on half reads its operands and writes its result through
// flush(x) = |x| < 2^-14 ? +-0 : x, as AGXCodeGenPrepare does for AGX
// (PTX-formalization half-0 k12: min.ftz.f16x2 kept a denormal).
bool isHalfTy(const Type *Ty) { return Ty->getScalarType()->isHalfTy(); }

bool isHalfArithmetic(const Instruction &I) {
  if (!isHalfTy(I.getType()))
    return false;
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return true;
  default:
    break;
  }
  const auto *CI = dyn_cast<CallInst>(&I);
  const Function *Callee = CI ? CI->getCalledFunction() : nullptr;
  if (!Callee)
    return false;
  switch (Callee->getIntrinsicID()) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::canonicalize:
    return true;
  case Intrinsic::not_intrinsic:
    break;
  default:
    return false;
  }
  // The portable passes' AIR math functions (air.fma.f16, air.fast_exp2.f16).
  StringRef Name = Callee->getName();
  if (!Name.consume_front("air."))
    return false;
  Name.consume_front("fast_") || Name.consume_front("precise_");
  Name = Name.take_until([](char C) { return C == '.'; });
  return StringSwitch<bool>(Name)
      .Cases({"fma", "fmax", "fmin", "fmax3", "fmin3", "sqrt", "rsqrt"}, true)
      .Cases({"exp", "exp2", "exp10", "log", "log2", "log10"}, true)
      .Cases({"sin", "cos", "tan", "floor", "ceil", "trunc", "rint"}, true)
      .Cases({"round", "divide", "recip", "pow", "powr", "fdim"}, true)
      .Default(false);
}

void flushHalfDenormals(Module &M) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    DenormalMode Mode = F.getDenormalMode(APFloat::IEEEhalf());
    bool FlushInputs = Mode.Input != DenormalMode::IEEE &&
                       Mode.Input != DenormalMode::Dynamic;
    bool FlushOutputs = Mode.Output != DenormalMode::IEEE &&
                        Mode.Output != DenormalMode::Dynamic;
    if (!FlushInputs && !FlushOutputs)
      continue;
    bool PositiveZero = Mode.Output == DenormalMode::PositiveZero;

    SmallPtrSet<Value *, 16> Flushed;
    auto Flush = [&](IRBuilder<> &B, Value *X, bool Positive) -> Value * {
      if (Flushed.contains(X))
        return X;
      if (auto *C = dyn_cast<ConstantFP>(X))
        if (!C->getValueAPF().isDenormal())
          return X;
      if (isa<Constant>(X) && !isa<ConstantFP>(X) &&
          !cast<Constant>(X)->containsUndefOrPoisonElement() &&
          !isa<ConstantExpr>(X)) {
        // A vector constant: flush only one with a denormal element.
        auto *CV = cast<Constant>(X);
        bool Denormal = false;
        if (auto *VT = dyn_cast<FixedVectorType>(CV->getType()))
          for (unsigned K = 0; K < VT->getNumElements(); ++K)
            if (auto *E = dyn_cast_or_null<ConstantFP>(
                    CV->getAggregateElement(K)))
              Denormal |= E->getValueAPF().isDenormal();
        if (!Denormal)
          return X;
      }
      Type *Ty = X->getType();
      Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
      Value *Small = B.CreateFCmpOLT(
          Abs, ConstantFP::get(Ty, APFloat::getSmallestNormalized(
                                       APFloat::IEEEhalf())));
      Value *Zero = Positive ? ConstantFP::get(Ty, 0.0)
                             : B.CreateBinaryIntrinsic(
                                   Intrinsic::copysign,
                                   ConstantFP::get(Ty, 0.0), X);
      Value *R = B.CreateSelect(Small, Zero, X);
      Flushed.insert(R);
      return R;
    };

    SmallVector<Instruction *, 32> Worklist;
    for (Instruction &I : instructions(F)) {
      Type *Op0 = I.getNumOperands() ? I.getOperand(0)->getType() : nullptr;
      bool Reads = isHalfArithmetic(I) ||
                   ((isa<FCmpInst>(I) || isa<FPExtInst>(I) ||
                     isa<FPToSIInst>(I) || isa<FPToUIInst>(I)) &&
                    isHalfTy(Op0));
      bool Writes = isa<FPTruncInst>(I) && isHalfTy(I.getType());
      if (Reads || Writes)
        Worklist.push_back(&I);
    }
    for (Instruction *I : Worklist) {
      IRBuilder<> B(I);
      bool Arith = isHalfArithmetic(*I);
      if (FlushInputs && !isa<FPTruncInst>(I)) {
        unsigned NumOps = isa<CallBase>(I) ? cast<CallBase>(I)->arg_size()
                                           : I->getNumOperands();
        for (unsigned Op = 0; Op < NumOps; ++Op)
          if (I->getOperand(Op)->getType()->isFPOrFPVectorTy() &&
              isHalfTy(I->getOperand(Op)->getType()))
            I->setOperand(Op, Flush(B, I->getOperand(Op), false));
      }
      if (auto *II = dyn_cast<IntrinsicInst>(I);
          II && II->getIntrinsicID() == Intrinsic::canonicalize) {
        Value *Op = II->getArgOperand(0);
        if (FlushOutputs && !Flushed.contains(Op))
          Op = Flush(B, Op, PositiveZero);
        II->replaceAllUsesWith(Op);
        II->eraseFromParent();
        continue;
      }
      if (FlushOutputs && (Arith || isa<FPTruncInst>(I))) {
        B.SetInsertPoint(I->getNextNode());
        SmallVector<User *, 4> OldUsers(I->users());
        Value *R = Flush(B, I, PositiveZero);
        I->replaceUsesWithIf(R, [&](Use &U) {
          return is_contained(OldUsers, U.getUser());
        });
      }
    }
  }
}

bool legalizeUnsupportedIR(Module &M) {
  genericToDevice(M);
  defineThreadgroupGlobals(M);
  stripLifetimeIntrinsics(M);
  expandIntrinsicsMetalMishandles(M);
  wideAIRFunctionsToIntrinsics(M);
  expandWideIntegers(M);
  gepsOfIntToPtrAsIntegers(M);
  flushHalfDenormals(M);
  lowerFreezeInsts(M);
  canonicalizeNNegZExt(M);
  stripDisjointFlags(M);
  lowerCmpIntrinsics(M);
  return true;
}

} // namespace

PreservedAnalyses
AIRLegalizeUnsupportedIRPass::run(Module &M, ModuleAnalysisManager &AM) {
  return legalizeUnsupportedIR(M) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

bool AIRLegalizeUnsupportedIRLegacy::runOnModule(Module &M) {
  return legalizeUnsupportedIR(M);
}

char AIRLegalizeUnsupportedIRLegacy::ID = 0;

INITIALIZE_PASS(AIRLegalizeUnsupportedIRLegacy, DEBUG_TYPE,
                "AIR Legalize Unsupported IR", false, false)

ModulePass *llvm::createAIRLegalizeUnsupportedIRLegacyPass() {
  return new AIRLegalizeUnsupportedIRLegacy();
}
