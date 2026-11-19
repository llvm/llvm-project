//===- WideningIntegerArithmetic.cpp - Widening Integer Arithmetic Pass
//----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the new Sign Extension Elimination pass.
/// First, widening integer solutions are computed for each Value of interest
/// They hold the fillType and other relevant fields.
/// The first part of the algorithm implemented here is described in the paper
/// "Widening Integer Arithmetic, Kevin Redwine and Norman Ramsey, CC 2004"
///
///
/// An overview of the whole algorithm:
///
/// First, we compute various solutions for a given value V
/// that represent the possible widths of V for example, (i16, i32, i64)
/// and for a given width W, we have a matching fillType
///
/// A fillType indicates for an operand what it produces and accepts in its
/// upper bits
///
/// An operand has a fillType only if it is widenable, i.e,
/// if by applying the operator to a wider value,
/// we can simulate the operator applied to narrow values.
///
/// wide value: [ fillType bits ][ original bits ]
///
/// SIGN: [ b b b b b b b ][ b . . . . . . . ]
/// ZEROS: [ 0 0 0 0 0 0 0 ][ . . . . . . . . ]
/// ANYTHING: [ ? ? ? ? ? ? ? ? ][ . . . . . . . . ]
///
/// For example since the OR instruction can produce a ZERO when operands are
/// zero-filled, 8-bit OR can be implemented as a 16-bit OR.
//
/// x narrow: 1010 0101
/// y narrow: 0001 1000
/// result narrow: 1011 1101

/// zext(x) to i16: 0000 0000 1010 0101
/// zext(y) to i16: 0000 0000 0001 1000
/// -------------------
/// or result i16: 0000 0000 1011 1101

/// This clearly explains why the low bits do not change, if we have a fillType.
/// Then we can simulate the narrow operator with a wider one.

/// In terms of using a different simpler definition for the fillType,
/// we know that operators represent the truth of mathematics.
/// So, if we use the KnownBits of LLVM and see that the upper bits of the
/// operands of the OR are ZERO, then we can infer that OR will produce zero in
/// its upper bits.
///
/// Secondly, after we are done finding solutions for every value V of interest
/// we create extension plans that express how are we going to
/// rewrite a particular graph.
/// The algorithm uses a simple worklist algorithm
/// to find a suitable fillType for each value in the worklist
/// for the given lattice we are examining.
///
/// An example lattice is given below
///
///
///             g[n+1]
///            /  |   \
///      z[n+1]   |   s[n+1]
///        |     g[n]    |
///        |    / .   \  |
///        |  /   .    \ |
///       z[n]    .     s[n]
///        .      .      .
///        .      .      .
///        .      .      .
///       z[1]   g[1]   s[1]
///
/// g: represents Anything
/// s: represents Sign
/// z: represents Zeros
///
/// If we have found more than one legal fillType during the graph build.
/// We aren't sure before building the graph which one will be legal at the end.
/// So we delay the choice for later. This is handled by the PromotionFillSolver
/// and the PromoteConstraintSolver.
///
/// In order to handle graphs with cycles ie, removing extensions
/// inside loops, the algorithm tries to propagate an available fillType
/// to the operands of an instruction, that checks if we have violation
/// on operands and users constraints.
///
/// Finally, we use the Rebuilder to rewrite the IR
/// and remove any extensions we have proved that are legal.
///

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/WideningIntegerArithmeticInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/MathExtras.h"

#include <memory>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

/// @brief
/// Holds the pending cleanups of the Allocator
/// otherwise we can get leaks.
class SolutionArena {
  struct PendingClean {
    void *Ptr;
    // function ref is not owning, so we can't use it here
    void (*Destroy)(void *);
  };

  BumpPtrAllocator Allocator;
  SmallVector<PendingClean, 16> PendingCleanups;

public:
  template <typename T, typename... Args> T *allocateSolution(Args &&...args) {
    void *Mem = Allocator.Allocate(sizeof(T), llvm::Align(alignof(T)));
    T *Sol = new (Mem) T(std::forward<Args>(args)...);
    // Save constructed items here otherwise we can get leaks
    PendingCleanups.push_back(PendingClean{
        Sol, [](void *Ptr) { std::destroy_at(static_cast<T *>(Ptr)); }});
    return Sol;
  }
  void Reset() {
    for (const PendingClean &Cleanup : llvm::reverse(PendingCleanups))
      Cleanup.Destroy(Cleanup.Ptr);

    PendingCleanups.clear();
    Allocator.Reset();
  }

  ~SolutionArena() { Reset(); }
};

template <typename T, typename... Args>
static T *allocateSolution(SolutionArena &Arena, Args &&...args) {
  return Arena.allocateSolution<T>(std::forward<Args>(args)...);
}

/// @brief This WIA class implements the type rules from the paper
/// Widening Integer Arithmetic, CC 2004, Kevin Redwine and Norman Ramsey
/// At the end we have for every instruction some solutions containing some
/// metadata The most useful ones are those two : (UpdatedWidth, fillType)
/// UpdatedWidth : the new width of the Instruction I.
/// FillType: Denotes how are the upper bits filled with that UpdatedWidth.
class WideningIntegerArithmetic {
  const DataLayout *DL = nullptr;

private:
  // For every Value *V, holds all the available solutions
  // that we calculate with the WIA class.
  DenseMap<Value *, SmallVector<WideningIntegerSolutionInfo *, 8>>
      AvailableSolutions;

  // Use a BumpPtr Allocator for efficiency when allocating WIA solutions.
  SolutionArena SolutionAllocator;

  // Identify Values or Instructions that are not of interest for this
  // pass. Their type should stay the same.
  bool shouldIgnoreValue(Value *V);

  // Check if we have at least one solution for the Value V.
  bool isPartiallySolved(Value *V);

  // Try to add the GeneratedSol of V into the list of all the Solutions for V.
  // For this to happen the GeneratedSol must not be redundant given
  // the IsRedundant method defined in the WIA header file.
  bool addNonRedundant(Value *V, WideningIntegerSolutionInfo *GeneratedSol);

  // Create constant solutions at various legal widths.
  bool visitConstant(ConstantInt *CI);

  // Create solutions for undef or poison at all widths.
  bool visitUndefOrPoison(Value *V);

  // Check if V has no integer ops, so that this value can be rejected
  // by joining a relevant region of interest.
  bool hasNonIntegerOps(Value *V);

  // Check if this Instr is a Extension or a Trunc.
  bool isExtOrTrunc(Value *Instr);

  // Check if this solution is of Trunc ty.
  bool isTruncKind(WideningIntegerSolutionInfo *Solution);

  // Check if Ty is of Ptr type.
  bool isPtrType(Type *Ty);

  // Get the width of invo's return type.
  int getCallInstWidth(CallBase *CB);

  // Get the scalar bits of that type.
  unsigned short getTypeScalarBits(Type *Ty);

  std::vector<unsigned short> IntegerSizes = {8, 16, 32, 64};

  // Return a default fillType for V, most of the time ANYTHING or Top.
  // This can later be refined to SIGN or ZERO.
  IntegerFillType getDefaultFillType(Value *V);

  // Checks if V has a appropriate type for this pass
  // Or this value is not of any interest at all.
  bool hasAppropriateType(Value *V);

  // Return the WIA kind for a specific opcode.
  WIAKind getKindForOpcode(unsigned Opcode);

  bool isNegativeCost(InstructionCost Cost);
};

} // end anonymous namespace

// For now, we only support power of scalar int types less equal than 64.
static bool isSupportedScalarIntType(Type *Ty) {
  if (!Ty->isIntegerTy())
    return false;
  unsigned Width = Ty->getIntegerBitWidth();
  return Width <= 64 && llvm::isPowerOf2_32(Width);
}

bool WideningIntegerArithmetic::addNonRedundant(
    Value *V, WideningIntegerSolutionInfo *GeneratedSol) {
  bool WasRedundant = false;
  SmallVector<WideningIntegerSolutionInfo *, 8> &Solutions =
      AvailableSolutions[V];
  if (Solutions.size() >= 8)
    return false;

  for (const auto *It = Solutions.begin(); It != Solutions.end();) {
    int IsRedundant = (*It)->IsRedudant((*GeneratedSol));
    if (IsRedundant == -1) { // GeneratedSol is redudant
      WasRedundant = true;
      ++It;
    } else if (IsRedundant == 1) { // Sol is redudant
      It = Solutions.erase(It);
    } else { // ret == 0 no redudant found
      ++It;
    }
  }
  if (!WasRedundant) {
    AvailableSolutions[V].push_back(GeneratedSol);
    return true;
  }
  return false;
}

bool WideningIntegerArithmetic::isPtrType(Type *Ty) {
  return Ty->isPointerTy();
}

bool WideningIntegerArithmetic::isPartiallySolved(Value *V) {
  return !AvailableSolutions[V].empty();
}

bool WideningIntegerArithmetic::visitConstant(ConstantInt *CI) {
  auto &Sols = AvailableSolutions[CI];
  if (!Sols.empty())
    return false;

  static constexpr unsigned PossibleWidths[] = {1, 8, 16, 32, 64};
  unsigned FillTypeWidth;
  IntegerFillType FillCI;
  if (CI->isNegative()) {
    FillCI = SIGN;
    FillTypeWidth = CI->getValue().getSignificantBits();
  } else {
    FillCI = ZEROS;
    FillTypeWidth = CI->getValue().getActiveBits();
  }
  InstructionCost ZeroCost(0);
  for (auto Width : PossibleWidths) {
    if ((FillCI == SIGN && CI->getValue().isSignedIntN(Width)) ||
        (FillCI == ZEROS && CI->getValue().isIntN(Width))) {
      auto *Sol = allocateSolution<WideningIntegerSolutionInfo>(
          SolutionAllocator, CONSTANT_INT_OPC, CONSTANT_INT_OPC, FillCI,
          FillTypeWidth, Width, Width, ZeroCost, WIAK_CONSTANT, CI);
      Sols.push_back(Sol);
    }
  }
  return false;
}

bool WideningIntegerArithmetic::isNegativeCost(InstructionCost Cost) {
  if (!Cost.isValid())
    return false;
  return Cost < 0;
}

bool WideningIntegerArithmetic::visitUndefOrPoison(Value *V) {
  if (isPartiallySolved(V)) {
    return false;
  }
  for (auto W : IntegerSizes) {
    auto *Sol = allocateSolution<WideningIntegerSolutionInfo>(
        SolutionAllocator, CONSTANT_INT_OPC, CONSTANT_INT_OPC, ANYTHING, W, W,
        W, 0, llvm::WIAK_UNKNOWN, V);
    AvailableSolutions[V].push_back(Sol);
  }
  return false;
}

bool WideningIntegerArithmetic::hasAppropriateType(Value *V) {
  auto *Ty = V->getType();
  if (auto *I = dyn_cast<Instruction>(V)) {
    switch (I->getOpcode()) {
    case Instruction::PtrToInt:
    case Instruction::InsertValue:
    case Instruction::Select:
      return isSupportedScalarIntType(Ty);
    case Instruction::FCmp:
      return false;
    case Instruction::ICmp: {
      auto *CmpI = cast<ICmpInst>(I);
      return isSupportedScalarIntType(CmpI->getOperand(0)->getType());
    }
    case Instruction::Ret: {
      auto *RI = cast<ReturnInst>(I);
      if (Value *RetVal = RI->getReturnValue())
        return isSupportedScalarIntType(RetVal->getType());
      return false;
    }
    case Instruction::FPToSI:
    case Instruction::FPToUI:
      return isSupportedScalarIntType(Ty);
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::IntToPtr:
    case Instruction::FPExt:
    case Instruction::FPTrunc:
    case Instruction::AddrSpaceCast:
      return false;
    case Instruction::BitCast: {
      auto *CI = cast<BitCastInst>(I);
      auto *Op0 = CI->getOperand(0);
      if (isa<ExtractValueInst>(Op0) || isa<ConstantInt>(Op0)) {
        return false;
      }
      return isSupportedScalarIntType(CI->getDestTy()) &&
             isSupportedScalarIntType(CI->getSrcTy());
    }
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt: {
      auto *CI = cast<CastInst>(I);
      auto *SrcTy = CI->getSrcTy();
      auto *DstTy = CI->getDestTy();
      auto *Op0 = CI->getOperand(0);
      if (isa<InsertElementInst>(Op0) || isa<ExtractValueInst>(Op0) ||
          isa<ConstantInt>(Op0) || isa<CallInst>(Op0) || isa<InvokeInst>(Op0)) {
        return false;
      }
      if (SrcTy->isVectorTy() || DstTy->isVectorTy()) {
        return false;
      }
      if (llvm::isPowerOf2_32(SrcTy->getIntegerBitWidth()) &&
          llvm::isPowerOf2_32(DstTy->getIntegerBitWidth())) {
        return isSupportedScalarIntType(SrcTy) &&
               isSupportedScalarIntType(DstTy) && hasAppropriateType(Op0);
      }
      return false;
    }
    case Instruction::Load:
      return isSupportedScalarIntType(Ty);
    case Instruction::Store: {
      auto *SI = cast<StoreInst>(I);
      return isSupportedScalarIntType(SI->getValueOperand()->getType());
    }
    case Instruction::Switch:
      return isSupportedScalarIntType(
          cast<SwitchInst>(I)->getCondition()->getType());
    case Instruction::UncondBr:
    case Instruction::CondBr:
      return true;

    default:
      break;
    }
  }

  if (Ty->isIntegerTy())
    return isSupportedScalarIntType(Ty);

  return false;
}

IntegerFillType WideningIntegerArithmetic::getDefaultFillType(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    unsigned Opcode = I->getOpcode();
    switch (Opcode) {
    case Instruction::SExt:
      return llvm::SIGN;
    case Instruction::ZExt:
      return llvm::ZEROS;
    }
  }
  return llvm::ANYTHING;
}

bool WideningIntegerArithmetic::hasNonIntegerOps(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    switch (I->getOpcode()) {
    case Instruction::Load:
    case Instruction::Select:
      return !V->getType()->isIntegerTy();

    case Instruction::Store:
      return !cast<StoreInst>(V)->getValueOperand()->getType()->isIntegerTy();

    case Instruction::ICmp:
    case Instruction::PtrToInt:
    case Instruction::FPToSI:
    case Instruction::FPToUI:
      return false;

    case Instruction::Call:
    case Instruction::Invoke:
      return !I->getType()->isIntegerTy();

    case Instruction::Switch: {
      auto *SI = cast<SwitchInst>(I);
      return !SI->getCondition()->getType()->isIntegerTy();
    }

    default:
      if (isExtOrTrunc(V)) {
        return false;
      }

      // Fallback: check operand types
      for (Value *Op : I->operand_values()) {
        if (!Op->getType()->isIntegerTy() ||
            Op->getType()->isFPOrFPVectorTy()) {
          return true;
        }
      }
      break;
    }
  }

  return false;
}

bool WideningIntegerArithmetic::shouldIgnoreValue(Value *V) {
  if (V->getType()->isFloatingPointTy()) {
    return true;
  }
  if (!hasAppropriateType(V)) {
    return true;
  }
  if (auto *I = dyn_cast<Instruction>(V)) {
    auto Opc = I->getOpcode();
    switch (Opc) {
    case Instruction::GetElementPtr:
      return true;
    case Instruction::UncondBr:
    case Instruction::CondBr:
      return false;
    default:
      return hasNonIntegerOps(I);
    }
  }
  return false;
}

int WideningIntegerArithmetic::getCallInstWidth(CallBase *CB) {
  Type *RetTy = CB->getFunctionType()->getReturnType();
  if (isPtrType(RetTy)) {
    return DL->getPointerTypeSizeInBits(RetTy);
  }
  if (!RetTy->isIntegerTy()) {
    return 0;
  }
  return RetTy->getIntegerBitWidth();
}

unsigned short WideningIntegerArithmetic::getTypeScalarBits(Type *Ty) {
  if (isPtrType(Ty)) {
    return DL->getPointerTypeSizeInBits(Ty);
  }
  if (Ty->isIntegerTy()) {
    return Ty->getScalarSizeInBits();
  }
  // for cmpxchg and others that have {i64, i1} type we just return the ptr size
  // in bits. since we do not touch those instructions.
  return DL->getPointerSizeInBits();
}

WIAKind WideningIntegerArithmetic::getKindForOpcode(unsigned Opcode) {
  switch (Opcode) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::Shl:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return WIAK_BINOP;
  case Instruction::ZExt:
  case Instruction::SExt:
    return WIAK_DROP_EXT;
  case Instruction::Trunc:
    return WIAK_NOOP;
  case Instruction::Load:
    return WIAK_LOAD;
  case Instruction::Store:
    return WIAK_STORE;
  case Instruction::PHI:
    return WIAK_PHI;
  case Instruction::Switch:
    return WIAK_SWITCH;
  case Instruction::Ret:
    return WIAK_RET;
  case Instruction::ICmp:
    return WIAK_ICMP;
  default:
    return WIAK_UNKNOWN;
  }
}

bool WideningIntegerArithmetic::isExtOrTrunc(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    unsigned Opc = I->getOpcode();
    return Opc == Instruction::ZExt || Opc == Instruction::SExt ||
           Opc == Instruction::Trunc;
  }
  return false;
}

bool WideningIntegerArithmetic::isTruncKind(
    WideningIntegerSolutionInfo *Solution) {
  if ((Solution->getKind() == llvm::WIAK_DROP_LOCOPY ||
       Solution->getKind() == llvm::WIAK_DROP_LOIGNORE))
    return true;

  return false;
}
