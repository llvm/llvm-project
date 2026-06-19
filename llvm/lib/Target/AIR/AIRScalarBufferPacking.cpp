//===- AIRScalarBufferPacking.cpp - Pack scalar params into one buffer --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRScalarBufferPacking.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#include <climits>
#include <string>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "air-scalar-buffer-packing"

namespace {
// AIR address space numbers.
constexpr unsigned ASDevice = 1;
constexpr unsigned ASConstant = 2;
} // namespace

static std::pair<unsigned, unsigned> scalarSizeAlign(Type *T) {
  if (T->isFloatTy())
    return {4, 4};
  if (T->isDoubleTy())
    return {8, 8};
  if (T->isHalfTy() || T->isBFloatTy())
    return {2, 2};
  if (auto *IT = dyn_cast<IntegerType>(T)) {
    unsigned Bits = IT->getBitWidth();
    if (Bits <= 8)
      return {1, 1};
    if (Bits <= 16)
      return {2, 2};
    if (Bits <= 32)
      return {4, 4};
    return {8, 8};
  }
  return {4, 4};
}

static Type *inferConstPtrLoadType(Function &F, unsigned ParamIdx) {
  Argument *Arg = F.getArg(ParamIdx);
  for (User *U : Arg->users())
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
  return nullptr;
}

static bool scalarBufferPacking(Module &M) {
  bool Changed = false;

  SmallPtrSet<Function *, 4> CalledFns;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction())
            CalledFns.insert(Callee);

  SmallVector<Function *, 4> Funcs;
  for (Function &F : M)
    if (!F.isDeclaration() && !CalledFns.count(&F))
      Funcs.push_back(&F);

  for (Function *FPtr : Funcs) {
    Function &F = *FPtr;

    if (F.getMetadata("air.argbuf.member_as"))
      continue;

    // Collect system value param indices from pre-baked metadata
    SmallDenseSet<unsigned, 4> SysValParams;
    if (auto *KMD = M.getNamedMetadata("air.kernel")) {
      for (unsigned k = 0; k < KMD->getNumOperands(); k++) {
        auto *Node = KMD->getOperand(k);
        if (Node->getNumOperands() < 1)
          continue;
        auto *FnMD = dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(0));
        if (!FnMD || FnMD->getValue() != &F)
          continue;
        for (unsigned n = 1; n < Node->getNumOperands(); n++) {
          auto *Sub = dyn_cast_if_present<MDNode>(Node->getOperand(n));
          if (!Sub)
            continue;
          for (unsigned s = 0; s < Sub->getNumOperands(); s++) {
            auto *ParamNode = dyn_cast_if_present<MDNode>(Sub->getOperand(s));
            if (!ParamNode || ParamNode->getNumOperands() < 2)
              continue;
            if (auto *Str = dyn_cast<MDString>(ParamNode->getOperand(1)))
              if (Str->getString().starts_with("air.thread") ||
                  Str->getString().starts_with("air.threadgroup"))
                if (auto *Idx =
                        dyn_cast<ConstantAsMetadata>(ParamNode->getOperand(0)))
                  SysValParams.insert(
                      cast<ConstantInt>(Idx->getValue())->getZExtValue());
          }
        }
      }
    }

    struct ScalarParam {
      unsigned OrigIdx;
      Type *ScalarType;
      bool IsConstPtr;
      bool IsDead;
    };
    SmallVector<ScalarParam, 8> ScalarParams;

    // Identify descriptor groups for type inference of dead params.
    //
    // ── Cross-component coupling (keep in sync with the Python driver) ──
    // Pattern-matches the flat descriptor layout produced by upstream
    // Triton's expand_signature. For a no-metadata descriptor this emits
    // [i64×(2N), i1, i1, i32×N, i64×N] where N = ndim, group_size = 4*N + 2.
    SmallDenseMap<unsigned, Type *, 16> DescriptorTypeForParam;
    {
      unsigned i = 0;
      while (i < F.arg_size()) {
        Type *T = F.getArg(i)->getType();
        bool IsDevicePtr =
            T->isPointerTy() && T->getPointerAddressSpace() == ASDevice;
        if (IsDevicePtr) {
          unsigned GroupStart = i + 1;
          unsigned j = GroupStart;
          while (j < F.arg_size()) {
            Type *Tj = F.getArg(j)->getType();
            if (!Tj->isPointerTy() ||
                Tj->getPointerAddressSpace() != ASConstant)
              break;
            j++;
          }
          unsigned Count = j - GroupStart;
          if (Count >= 2 && (Count - 2) % 4 == 0) {
            unsigned Ndim = (Count - 2) / 4;
            auto GetPatternType = [&](unsigned Offset) -> Type * {
              if (Offset < 2 * Ndim)
                return Type::getInt64Ty(M.getContext());
              if (Offset < 2 * Ndim + 2)
                return Type::getInt1Ty(M.getContext());
              if (Offset < 3 * Ndim + 2)
                return Type::getInt32Ty(M.getContext());
              return Type::getInt64Ty(M.getContext());
            };
            bool Valid = true;
            for (unsigned Off = 0; Off < Count; Off++) {
              Type *LiveTy = inferConstPtrLoadType(F, GroupStart + Off);
              if (LiveTy) {
                auto [LiveSize, _a] = scalarSizeAlign(LiveTy);
                auto [PatSize, _b] = scalarSizeAlign(GetPatternType(Off));
                if (LiveSize != PatSize) {
                  Valid = false;
                  break;
                }
              }
            }
            if (Valid) {
              for (unsigned Off = 0; Off < Count; Off++)
                DescriptorTypeForParam[GroupStart + Off] = GetPatternType(Off);
            }
          }
        }
        i++;
      }
    }

    for (unsigned i = 0; i < F.arg_size(); i++) {
      if (SysValParams.count(i))
        continue;
      Type *T = F.getArg(i)->getType();
      if (T->isPointerTy()) {
        if (T->getPointerAddressSpace() == ASConstant) {
          Type *LoadTy = inferConstPtrLoadType(F, i);
          if (LoadTy && LoadTy->isPointerTy())
            LoadTy = Type::getInt64Ty(M.getContext());
          if (LoadTy) {
            ScalarParams.push_back({i, LoadTy, true, false});
          } else {
            Type *DeadTy = Type::getInt32Ty(M.getContext());
            auto It = DescriptorTypeForParam.find(i);
            if (It != DescriptorTypeForParam.end())
              DeadTy = It->second;
            ScalarParams.push_back({i, DeadTy, true, true});
          }
        }
        continue;
      }
      if (T->isVectorTy() || T->isAggregateType())
        continue;
      ScalarParams.push_back({i, T, false, false});
    }

    if (ScalarParams.empty())
      continue;

    // Compute byte offsets (matches Python _compute_scalar_layout).
    SmallVector<unsigned, 8> FieldOffsets;
    unsigned CurrentOffset = 0;
    for (auto &SP : ScalarParams) {
      auto [Size, Align_] = scalarSizeAlign(SP.ScalarType);
      unsigned Padding = (Align_ - (CurrentOffset % Align_)) % Align_;
      CurrentOffset += Padding;
      FieldOffsets.push_back(CurrentOffset);
      CurrentOffset += Size;
    }

    // Use float as the GEP/load element type for typed-pointer consistency
    // (AIR v1 bitcode rejects load type != pointee type).
    Type *BufElemTy = Type::getFloatTy(M.getContext());
    unsigned BufElemSize = 4;

    SmallDenseSet<unsigned, 8> ScalarIdxSet;
    for (auto &SP : ScalarParams)
      ScalarIdxSet.insert(SP.OrigIdx);

    SmallVector<Type *, 8> NewParamTypes;
    SmallVector<unsigned, 8> OldToNew(F.arg_size(), UINT_MAX);
    unsigned BufNewIdx = UINT_MAX;
    unsigned Ni = 0;
    for (unsigned i = 0; i < F.arg_size(); i++) {
      if (ScalarIdxSet.count(i))
        continue;
      OldToNew[i] = Ni;
      NewParamTypes.push_back(F.getArg(i)->getType());
      Ni++;
    }
    BufNewIdx = Ni;
    NewParamTypes.push_back(PointerType::get(M.getContext(), metal::AS::Device));
    Ni++;

    SmallVector<std::string, 8> OldArgNames;
    for (unsigned i = 0; i < F.arg_size(); i++)
      OldArgNames.push_back(F.getArg(i)->getName().str());

    auto *NewFTy = FunctionType::get(F.getFunctionType()->getReturnType(),
                                     NewParamTypes, false);
    auto *NewF =
        Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", &M);
    NewF->copyAttributesFrom(&F);
    NewF->copyMetadata(&F, 0);
    NewF->splice(NewF->begin(), &F);

    for (unsigned i = 0; i < F.arg_size(); i++) {
      if (ScalarIdxSet.count(i))
        continue;
      Argument *OldArg = F.getArg(i);
      Argument *NewArg = NewF->getArg(OldToNew[i]);
      NewArg->setName(OldArgNames[i]);
      OldArg->replaceAllUsesWith(NewArg);
    }

    Argument *BufArg = NewF->getArg(BufNewIdx);
    BufArg->setName("_scalar_buf");

    BasicBlock &EntryBB = NewF->getEntryBlock();
    SmallVector<Instruction *, 16> Preamble;

    for (unsigned j = 0; j < ScalarParams.size(); j++) {
      auto &SP = ScalarParams[j];
      Argument *OldArg = F.getArg(SP.OrigIdx);
      std::string Name = OldArgNames[SP.OrigIdx];
      if (SP.IsConstPtr && Name.size() > 4 &&
          Name.substr(Name.size() - 4) == "_ptr")
        Name = Name.substr(0, Name.size() - 4);

      if (SP.IsDead) {
        if (!OldArg->use_empty())
          OldArg->replaceAllUsesWith(UndefValue::get(OldArg->getType()));
        continue;
      }

      unsigned GepIdx = FieldOffsets[j] / BufElemSize;

      auto *Gep = GetElementPtrInst::CreateInBounds(
          BufElemTy, BufArg,
          ConstantInt::get(Type::getInt64Ty(M.getContext()), GepIdx),
          Name + "_gep");
      Preamble.push_back(Gep);

      Value *Loaded;
      if (SP.ScalarType == BufElemTy) {
        auto *Ld = new LoadInst(BufElemTy, Gep, Name, false, Align(4));
        Preamble.push_back(Ld);
        Loaded = Ld;
      } else {
        unsigned ScalarBits = SP.ScalarType->getScalarSizeInBits();
        if (SP.ScalarType->isPointerTy()) {
          // Pointer: load two i32 words, combine to i64, then inttoptr.
          auto *RawLoad =
              new LoadInst(BufElemTy, Gep, Name + "_raw", false, Align(4));
          Preamble.push_back(RawLoad);
          auto *LoI32 = CastInst::Create(Instruction::BitCast, RawLoad,
                                         Type::getInt32Ty(M.getContext()),
                                         Name + "_lo32");
          Preamble.push_back(LoI32);
          auto *LoI64 = CastInst::Create(Instruction::ZExt, LoI32,
                                         Type::getInt64Ty(M.getContext()),
                                         Name + "_lo64");
          Preamble.push_back(LoI64);

          auto *GepHi = GetElementPtrInst::CreateInBounds(
              BufElemTy, BufArg,
              ConstantInt::get(Type::getInt64Ty(M.getContext()), GepIdx + 1),
              Name + "_gep_hi");
          Preamble.push_back(GepHi);
          auto *HiRaw =
              new LoadInst(BufElemTy, GepHi, Name + "_hi_raw", false, Align(4));
          Preamble.push_back(HiRaw);
          auto *HiI32 = CastInst::Create(Instruction::BitCast, HiRaw,
                                         Type::getInt32Ty(M.getContext()),
                                         Name + "_hi32");
          Preamble.push_back(HiI32);
          auto *HiI64 = CastInst::Create(Instruction::ZExt, HiI32,
                                         Type::getInt64Ty(M.getContext()),
                                         Name + "_hi64");
          Preamble.push_back(HiI64);
          auto *Shifted = BinaryOperator::Create(
              Instruction::Shl, HiI64,
              ConstantInt::get(Type::getInt64Ty(M.getContext()), 32),
              Name + "_shift");
          Preamble.push_back(Shifted);
          auto *Combined = BinaryOperator::Create(Instruction::Or, Shifted,
                                                  LoI64, Name + "_i64");
          Preamble.push_back(Combined);
          auto *Ptr = CastInst::Create(Instruction::IntToPtr, Combined,
                                       SP.ScalarType, Name);
          Preamble.push_back(Ptr);
          Loaded = Ptr;
        } else if (ScalarBits == 32) {
          auto *RawLoad =
              new LoadInst(BufElemTy, Gep, Name + "_raw", false, Align(4));
          Preamble.push_back(RawLoad);
          auto *Cast = CastInst::Create(Instruction::BitCast, RawLoad,
                                        SP.ScalarType, Name);
          Preamble.push_back(Cast);
          Loaded = Cast;
        } else {
          auto *RawLoad =
              new LoadInst(BufElemTy, Gep, Name + "_raw", false, Align(4));
          Preamble.push_back(RawLoad);
          auto *AsI32 =
              CastInst::Create(Instruction::BitCast, RawLoad,
                               Type::getInt32Ty(M.getContext()), Name + "_i32");
          Preamble.push_back(AsI32);
          if (SP.ScalarType->isIntegerTy()) {
            unsigned TargetBits = SP.ScalarType->getIntegerBitWidth();
            if (TargetBits > 32) {
              // i64: combine two float loads (lo, hi).
              auto *LoI32 = AsI32;
              auto *LoI64 = CastInst::Create(Instruction::ZExt, LoI32,
                                             Type::getInt64Ty(M.getContext()),
                                             Name + "_lo64");
              Preamble.push_back(LoI64);

              auto *GepHi = GetElementPtrInst::CreateInBounds(
                  BufElemTy, BufArg,
                  ConstantInt::get(Type::getInt64Ty(M.getContext()),
                                   GepIdx + 1),
                  Name + "_gep_hi");
              Preamble.push_back(GepHi);
              auto *HiRaw = new LoadInst(BufElemTy, GepHi, Name + "_hi_raw",
                                         false, Align(4));
              Preamble.push_back(HiRaw);
              auto *HiI32 = CastInst::Create(Instruction::BitCast, HiRaw,
                                             Type::getInt32Ty(M.getContext()),
                                             Name + "_hi32");
              Preamble.push_back(HiI32);
              auto *HiI64 = CastInst::Create(Instruction::ZExt, HiI32,
                                             Type::getInt64Ty(M.getContext()),
                                             Name + "_hi64");
              Preamble.push_back(HiI64);

              auto *Shifted = BinaryOperator::Create(
                  Instruction::Shl, HiI64,
                  ConstantInt::get(Type::getInt64Ty(M.getContext()), 32),
                  Name + "_shift");
              Preamble.push_back(Shifted);
              auto *Combined =
                  BinaryOperator::Create(Instruction::Or, Shifted, LoI64, Name);
              Preamble.push_back(Combined);
              Loaded = Combined;
            } else {
              Instruction::CastOps Op;
              if (TargetBits < 32)
                Op = Instruction::Trunc;
              else
                Op = Instruction::BitCast;
              auto *Conv = CastInst::Create(Op, AsI32, SP.ScalarType, Name);
              Preamble.push_back(Conv);
              Loaded = Conv;
            }
          } else if (SP.ScalarType->isHalfTy() || SP.ScalarType->isBFloatTy()) {
            auto *AsI16 = CastInst::Create(Instruction::Trunc, AsI32,
                                           Type::getInt16Ty(M.getContext()),
                                           Name + "_i16");
            Preamble.push_back(AsI16);
            auto *Cast = CastInst::Create(Instruction::BitCast, AsI16,
                                          SP.ScalarType, Name);
            Preamble.push_back(Cast);
            Loaded = Cast;
          } else if (SP.ScalarType->isDoubleTy()) {
            auto *LoI64 = CastInst::Create(Instruction::ZExt, AsI32,
                                           Type::getInt64Ty(M.getContext()),
                                           Name + "_lo64");
            Preamble.push_back(LoI64);
            auto *GepHi = GetElementPtrInst::CreateInBounds(
                BufElemTy, BufArg,
                ConstantInt::get(Type::getInt64Ty(M.getContext()), GepIdx + 1),
                Name + "_gep_hi");
            Preamble.push_back(GepHi);
            auto *HiRaw = new LoadInst(BufElemTy, GepHi, Name + "_hi_raw", false,
                                       Align(4));
            Preamble.push_back(HiRaw);
            auto *HiI32 = CastInst::Create(Instruction::BitCast, HiRaw,
                                           Type::getInt32Ty(M.getContext()),
                                           Name + "_hi32");
            Preamble.push_back(HiI32);
            auto *HiI64 = CastInst::Create(Instruction::ZExt, HiI32,
                                           Type::getInt64Ty(M.getContext()),
                                           Name + "_hi64");
            Preamble.push_back(HiI64);
            auto *Shifted = BinaryOperator::Create(
                Instruction::Shl, HiI64,
                ConstantInt::get(Type::getInt64Ty(M.getContext()), 32),
                Name + "_shift");
            Preamble.push_back(Shifted);
            auto *Combined = BinaryOperator::Create(Instruction::Or, Shifted,
                                                    LoI64, Name + "_bits");
            Preamble.push_back(Combined);
            auto *Cast = CastInst::Create(Instruction::BitCast, Combined,
                                          SP.ScalarType, Name);
            Preamble.push_back(Cast);
            Loaded = Cast;
          } else {
            auto *Conv = CastInst::Create(Instruction::BitCast, AsI32,
                                          SP.ScalarType, Name);
            Preamble.push_back(Conv);
            Loaded = Conv;
          }
        }
      }

      Value *Load = Loaded;

      if (SP.IsConstPtr) {
        SmallVector<LoadInst *, 4> OldLoads;
        for (User *U : OldArg->users())
          if (auto *LI = dyn_cast<LoadInst>(U))
            OldLoads.push_back(LI);
        for (LoadInst *LI : OldLoads) {
          LI->replaceAllUsesWith(Load);
          LI->eraseFromParent();
        }
        if (!OldArg->use_empty())
          OldArg->replaceAllUsesWith(UndefValue::get(OldArg->getType()));
      } else {
        OldArg->replaceAllUsesWith(Load);
      }
    }

    // Insert preamble at entry in original order.
    for (Instruction *I : llvm::reverse(Preamble))
      I->insertBefore(EntryBB, EntryBB.begin());

    // Update metadata references to point at the new function.
    for (NamedMDNode &NMD : M.named_metadata())
      for (unsigned i = 0; i < NMD.getNumOperands(); i++) {
        MDNode *Node = NMD.getOperand(i);
        for (unsigned j = 0; j < Node->getNumOperands(); j++)
          if (auto *VMD =
                  dyn_cast_if_present<ValueAsMetadata>(Node->getOperand(j)))
            if (VMD->getValue() == &F)
              Node->replaceOperandWith(j, ValueAsMetadata::get(NewF));
      }

    std::string FName = F.getName().str();
    F.eraseFromParent();
    NewF->setName(FName);
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses AIRScalarBufferPackingPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  return scalarBufferPacking(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

bool AIRScalarBufferPackingLegacy::runOnModule(Module &M) {
  return scalarBufferPacking(M);
}

char AIRScalarBufferPackingLegacy::ID = 0;

INITIALIZE_PASS(AIRScalarBufferPackingLegacy, DEBUG_TYPE,
                "AIR Scalar Buffer Packing", false, false)

ModulePass *llvm::createAIRScalarBufferPackingLegacyPass() {
  return new AIRScalarBufferPackingLegacy();
}
