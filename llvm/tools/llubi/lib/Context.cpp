//===- Context.cpp - State Tracking for llubi -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tracks the global states (e.g., memory) of the interpreter.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace llvm::ubi {

Context::Context(Module &M, const AsmParserContext *ParserContext)
    : Ctx(M.getContext()), M(M), ParserContext(ParserContext),
      DL(M.getDataLayout()), TLIImpl(M.getTargetTriple()) {}

Context::~Context() = default;

bool Context::initGlobalValues() {
  // Register all function and block targets that may be used by indirect calls
  // and branches.
  for (Function &F : M) {
    if (F.hasAddressTaken()) {
      // TODO: Use precise alignment for function pointers if it is necessary.
      auto FuncObj = allocate(0, F.getPointerAlignment(DL).value(), F.getName(),
                              DL.getProgramAddressSpace(), MemInitKind::Zeroed,
                              MemAllocKind::Global, /*IsIRGlobalValue=*/true);
      if (!FuncObj)
        return false;
      ValidFuncTargets.try_emplace(FuncObj->getAddress(),
                                   std::make_pair(&F, FuncObj));
      FuncAddrMap.try_emplace(&F, deriveFromMemoryObject(FuncObj));
    }

    for (BasicBlock &BB : F) {
      if (!BB.hasAddressTaken())
        continue;
      auto BlockObj = allocate(0, 1, BB.getName(), DL.getProgramAddressSpace(),
                               MemInitKind::Zeroed, MemAllocKind::BlockAddress);
      if (!BlockObj)
        return false;
      ValidBlockTargets.try_emplace(BlockObj->getAddress(),
                                    std::make_pair(&BB, BlockObj));
      BlockAddrMap.try_emplace(&BB, deriveFromMemoryObject(BlockObj));
    }
  }

  for (GlobalVariable &GV : M.globals()) {
    Type *ValueTy = GV.getValueType();
    const uint64_t Size = getEffectiveTypeAllocSize(ValueTy);
    Align Alignment = GV.getPointerAlignment(DL);
    auto InitKind =
        GV.hasInitializer() ? MemInitKind::Zeroed : MemInitKind::Uninitialized;
    const auto Obj =
        allocate(Size, Alignment.value(), GV.getName(), GV.getAddressSpace(),
                 InitKind, MemAllocKind::Global, /*IsIRGlobalValue=*/true);

    if (!Obj)
      return false;

    Obj->setIsConstant(GV.isConstant());
    GlobalAddrMap.try_emplace(&GV, deriveFromMemoryObject(Obj));
  }

  for (GlobalVariable &GV : M.globals()) {
    if (!GV.hasInitializer())
      continue;

    MemoryObject *Obj = GlobalAddrMap.at(&GV).provenance().getMemoryObject();
    assert(Obj && "global pointer should have memory object provenance");

    Constant *Init = GV.getInitializer();

    const AnyValue *InitVal = getConstantValue(Init);
    if (!InitVal)
      return false;

    store(*Obj, 0, *InitVal, GV.getValueType());
    resetNoncacheableConstantBuffer();
  }
  return true;
}

MaterializedConstant Context::getConstantValueImpl(Constant *C) {
  if (isa<PoisonValue>(C))
    return MaterializedConstant(AnyValue::getPoisonValue(*this, C->getType()),
                                /*Cacheable=*/true);

  if (isa<UndefValue>(C)) {
    // We treat undef as a freshly freeze poison.
    auto Value = AnyValue::getPoisonValue(*this, C->getType());
    freeze(Value, C->getType());
    return MaterializedConstant(std::move(Value), /*Cacheable=*/false);
  }

  if (isa<ConstantAggregateZero>(C))
    return MaterializedConstant(AnyValue::getNullValue(*this, C->getType()),
                                /*Cacheable=*/true);

  if (isa<ConstantPointerNull>(C))
    return MaterializedConstant(AnyValue::getNullValue(*this, C->getType()),
                                /*Cacheable=*/true);

  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    if (auto *VecTy = dyn_cast<VectorType>(CI->getType()))
      return MaterializedConstant(
          std::vector<AnyValue>(getEVL(VecTy->getElementCount()),
                                AnyValue(CI->getValue())),
          /*Cacheable=*/true);
    return MaterializedConstant(CI->getValue(), /*Cacheable=*/true);
  }

  if (auto *CFP = dyn_cast<ConstantFP>(C)) {
    if (auto *VecTy = dyn_cast<VectorType>(CFP->getType()))
      return MaterializedConstant(
          std::vector<AnyValue>(getEVL(VecTy->getElementCount()),
                                AnyValue(CFP->getValue())),
          /*Cacheable=*/true);
    return MaterializedConstant(CFP->getValue(), /*Cacheable=*/true);
  }

  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    std::vector<AnyValue> Elts;
    Elts.reserve(CDS->getNumElements());
    bool Cacheable = true;
    for (uint32_t I = 0, E = CDS->getNumElements(); I != E; ++I) {
      auto Elt = getConstantValue(CDS->getElementAsConstant(I));
      if (!Elt)
        return std::nullopt;
      Cacheable &= Elt->isCacheable();
      Elts.push_back(*Elt);
    }
    return MaterializedConstant(std::move(Elts), Cacheable);
  }

  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    std::vector<AnyValue> Elts;
    Elts.reserve(CA->getNumOperands());
    bool Cacheable = true;
    for (uint32_t I = 0, E = CA->getNumOperands(); I != E; ++I) {
      auto Elt = getConstantValue(CA->getOperand(I));
      if (!Elt)
        return std::nullopt;
      Cacheable &= Elt->isCacheable();
      Elts.push_back(*Elt);
    }
    return MaterializedConstant(std::move(Elts), Cacheable);
  }

  if (auto *BA = dyn_cast<BlockAddress>(C))
    return MaterializedConstant(BlockAddrMap.at(BA->getBasicBlock()),
                                /*Cacheable=*/true);

  if (auto *GV = dyn_cast<GlobalVariable>(C))
    return MaterializedConstant(GlobalAddrMap.at(GV), /*Cacheable=*/true);

  if (auto *F = dyn_cast<Function>(C))
    return MaterializedConstant(FuncAddrMap.at(F), /*Cacheable=*/true);

  if (auto *CE = dyn_cast<ConstantExpr>(C))
    return evaluateConstantExpression(CE);

  return std::nullopt;
}

MaterializedConstant Context::evaluateConstantExpression(ConstantExpr *CE) {
  unsigned Opc = CE->getOpcode();
  switch (Opc) {
  case Instruction::Trunc: {
    const auto *Src = getConstantValue(CE->getOperand(0));
    if (!Src)
      return std::nullopt;
    if (Src->isPoison())
      return MaterializedConstant(AnyValue::poison(), Src->isCacheable());
    unsigned BitWidth = CE->getType()->getScalarSizeInBits();
    if (Src->isInteger())
      return MaterializedConstant(Src->asInteger().trunc(BitWidth),
                                  Src->isCacheable());
    std::vector<AnyValue> Vec = Src->asAggregate();
    for (auto &V : Vec) {
      if (V.isInteger())
        V = V.asInteger().trunc(BitWidth);
    }
    return MaterializedConstant(std::move(Vec), Src->isCacheable());
  }
  case Instruction::BitCast: {
    Constant *SrcOp = CE->getOperand(0);
    const auto *Src = getConstantValue(SrcOp);
    if (!Src)
      return std::nullopt;
    SmallVector<Byte> Bytes;
    Bytes.resize(getEffectiveTypeStoreSize(CE->getType()), Byte::concrete(0));
    toBytes(*Src, SrcOp->getType(), Bytes);
    return MaterializedConstant(fromBytes(Bytes, CE->getType()),
                                Src->isCacheable());
  }
  case Instruction::InsertElement: {
    const auto *Src = getConstantValue(CE->getOperand(0));
    if (!Src)
      return std::nullopt;
    const auto *Val = getConstantValue(CE->getOperand(1));
    if (!Val)
      return std::nullopt;
    const auto *Idx = getConstantValue(CE->getOperand(2));
    if (!Idx)
      return std::nullopt;
    auto &SrcVec = Src->asAggregate();
    bool Cacheable =
        Src->isCacheable() && Val->isCacheable() && Idx->isCacheable();
    if (Idx->isPoison() || Idx->asInteger().uge(SrcVec.size()))
      return MaterializedConstant(
          AnyValue::getPoisonValue(*this, CE->getType()), Cacheable);
    std::vector<AnyValue> ResVec = SrcVec;
    ResVec[Idx->asInteger().getZExtValue()] = *Val;
    return MaterializedConstant(std::move(ResVec), Cacheable);
  }
  case Instruction::ExtractElement: {
    const auto *Src = getConstantValue(CE->getOperand(0));
    if (!Src)
      return std::nullopt;
    const auto *Idx = getConstantValue(CE->getOperand(1));
    if (!Idx)
      return std::nullopt;
    auto &SrcVec = Src->asAggregate();
    bool Cacheable = Src->isCacheable() && Idx->isCacheable();
    if (Idx->isPoison() || Idx->asInteger().uge(SrcVec.size()))
      return MaterializedConstant(
          AnyValue::getPoisonValue(*this, CE->getType()), Cacheable);
    return MaterializedConstant(SrcVec[Idx->asInteger().getZExtValue()],
                                Cacheable);
  }
  case Instruction::ShuffleVector: {
    const auto *LHS = getConstantValue(CE->getOperand(0));
    if (!LHS)
      return std::nullopt;
    const auto *RHS = getConstantValue(CE->getOperand(1));
    if (!RHS)
      return std::nullopt;
    auto &LHSVec = LHS->asAggregate();
    auto &RHSVec = RHS->asAggregate();
    uint32_t Size = cast<VectorType>(CE->getOperand(0)->getType())
                        ->getElementCount()
                        .getKnownMinValue();
    std::vector<AnyValue> Res;
    uint32_t DstLen =
        getEVL(cast<VectorType>(CE->getType())->getElementCount());
    Res.reserve(DstLen);
    uint32_t Stride = CE->getShuffleMask().size();
    // For scalable vectors, we need to repeat the shuffle mask until we fill
    // the destination vector.
    for (uint32_t Off = 0; Off != DstLen; Off += Stride) {
      for (int Idx : CE->getShuffleMask()) {
        if (Idx == PoisonMaskElem)
          Res.push_back(AnyValue::poison());
        else if (Idx < static_cast<int>(Size))
          Res.push_back(LHSVec[Idx]);
        else
          Res.push_back(RHSVec[Idx - Size]);
      }
    }
    return MaterializedConstant(std::move(Res),
                                LHS->isCacheable() && RHS->isCacheable());
  }
  case Instruction::GetElementPtr: {
    // Temporary variable for reference to poison values when the subexpression
    // cannot be evaluated. As the reference will be consumed immediately, we
    // don't need to store them into a list.
    AnyValue PoisonValue;
    bool Cacheable = true;
    AnyValue Res =
        computeGEP(*cast<GEPOperator>(CE), [&](Value *V) -> const AnyValue & {
          const auto *Val = getConstantValue(cast<Constant>(V));
          if (Val) {
            Cacheable &= Val->isCacheable();
            return *Val;
          }
          PoisonValue = AnyValue::getPoisonValue(*this, V->getType());
          return PoisonValue;
        });
    if (!PoisonValue.isNone())
      return std::nullopt;
    return MaterializedConstant(std::move(Res), Cacheable);
  }
  case Instruction::PtrToAddr:
  case Instruction::PtrToInt: {
    const auto *Src = getConstantValue(CE->getOperand(0));
    if (!Src)
      return std::nullopt;
    bool Cacheable = Opc == Instruction::PtrToAddr && Src->isCacheable();
    if (Src->isPoison())
      return MaterializedConstant(AnyValue::poison(), Cacheable);
    unsigned BitWidth = CE->getType()->getScalarSizeInBits();
    if (Src->isPointer()) {
      if (Opc == Instruction::PtrToInt)
        exposeProvenance(Src->asPointer().provenance());
      return MaterializedConstant(Src->asPointer().address().trunc(BitWidth),
                                  Cacheable);
    }
    std::vector<AnyValue> Vec = Src->asAggregate();
    for (auto &V : Vec) {
      if (V.isPointer()) {
        if (Opc == Instruction::PtrToInt)
          exposeProvenance(V.asPointer().provenance());
        V = V.asPointer().address().trunc(BitWidth);
      }
    }
    return MaterializedConstant(std::move(Vec), Cacheable);
  }
  case Instruction::IntToPtr: {
    const auto *Src = getConstantValue(CE->getOperand(0));
    if (!Src)
      return std::nullopt;
    if (Src->isPoison())
      return MaterializedConstant(AnyValue::poison(), /*Cacheable=*/false);
    unsigned BitWidth =
        DL.getPointerSizeInBits(CE->getType()->getPointerAddressSpace());
    if (Src->isInteger())
      return MaterializedConstant(
          Pointer(getWildcardProvenance(),
                  Src->asInteger().zextOrTrunc(BitWidth)),
          /*Cacheable=*/false);
    std::vector<AnyValue> Vec = Src->asAggregate();
    for (auto &V : Vec) {
      if (V.isInteger())
        V = Pointer(getWildcardProvenance(),
                    V.asInteger().zextOrTrunc(BitWidth));
    }
    return MaterializedConstant(std::move(Vec), /*Cacheable=*/false);
  }
  case Instruction::AddrSpaceCast:
    return std::nullopt;
  default:
    assert(Instruction::isBinaryOp(Opc) && "Must be binary operator?");
    const auto *LHS = getConstantValue(CE->getOperand(0));
    if (!LHS)
      return std::nullopt;
    const auto *RHS = getConstantValue(CE->getOperand(1));
    if (!RHS)
      return std::nullopt;

    bool HasNUW = false;
    bool HasNSW = false;
    if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(CE)) {
      HasNUW = OBO->hasNoUnsignedWrap();
      HasNSW = OBO->hasNoSignedWrap();
    }

    auto ScalarEval = [&](const AnyValue &LHS,
                          const AnyValue &RHS) -> AnyValue {
      if (LHS.isPoison() || RHS.isPoison())
        return AnyValue::poison();
      auto &LHSVal = LHS.asInteger();
      auto &RHSVal = RHS.asInteger();
      switch (Opc) {
      case Instruction::Add:
        return addNoWrap(LHSVal, RHSVal, HasNSW, HasNUW);
      case Instruction::Sub:
        return subNoWrap(LHSVal, RHSVal, HasNSW, HasNUW);
      case Instruction::Xor:
        return LHSVal ^ RHSVal;
      default:
        llvm_unreachable("Unsupported opcode in constant expression.");
      }
    };

    bool Cacheable = LHS->isCacheable() && RHS->isCacheable();

    if (CE->getType()->isVectorTy()) {
      auto &LHSVec = LHS->asAggregate();
      auto &RHSVec = RHS->asAggregate();
      std::vector<AnyValue> ResVec;
      ResVec.reserve(LHSVec.size());
      for (const auto &[ScalarLHS, ScalarRHS] : zip(LHSVec, RHSVec))
        ResVec.push_back(ScalarEval(ScalarLHS, ScalarRHS));
      return MaterializedConstant(std::move(ResVec), Cacheable);
    }

    return MaterializedConstant(ScalarEval(*LHS, *RHS), Cacheable);
  }
}

const MaterializedConstant *Context::getConstantValue(Constant *C) {
  auto It = ConstCache.find(C);
  if (It != ConstCache.end())
    return &It->second;

  MaterializedConstant Val = getConstantValueImpl(C);
  if (Val.isNone())
    return nullptr;
  if (!Val.isCacheable()) {
    assert(NoncacheableConstCount <= 1024 && "Unbounded temporary buffer.");
    ++NoncacheableConstCount;
    return new (NoncacheableConstBuffer.Allocate())
        MaterializedConstant(std::move(Val));
  }

  return &ConstCache.emplace(C, std::move(Val)).first->second;
}

void Context::resetNoncacheableConstantBuffer() {
  NoncacheableConstBuffer.DestroyAll();
  NoncacheableConstCount = 0;
}

APInt Context::getTag(uint32_t BitWidth, Provenance &Prov) {
  // Nullary provenance.
  if (!Prov.getMemoryObject())
    return APInt::getZero(BitWidth);
  // The tag is already initialized.
  if (!Prov.getTag().isZero())
    return Prov.getTag();

  // FIXME: This doesn't work when the address space is too small.
  while (true) {
    APInt Tag = generateRandomAPInt(BitWidth);
    if (Tag.isZero() || !TaggedProvenances.try_emplace(Tag, &Prov).second)
      continue;
    Prov.setTag(Tag);
    Prov.getMemoryObject()->AssociatedTags.push_back(Tag);
    return Tag;
  }
}

AnyValue Context::fromBytes(ConstBytesView Bytes, Type *Ty,
                            uint32_t OffsetInBits, bool CheckPaddingBits,
                            bool *ContainsUndefinedBits) {
  uint32_t NumBits = DL.getTypeSizeInBits(Ty).getFixedValue();
  uint32_t NewOffsetInBits = OffsetInBits + NumBits;
  if (CheckPaddingBits)
    NewOffsetInBits = alignTo(NewOffsetInBits, 8);
  bool NeedsPadding = NewOffsetInBits != OffsetInBits + NumBits;
  uint32_t NumBitsToExtract = NewOffsetInBits - OffsetInBits;
  uint32_t NumWords = APInt::getNumWords(NumBitsToExtract);
  constexpr uint32_t WordBits = APInt::APINT_BITS_PER_WORD;
  SmallVector<APInt::WordType> RawBits(NumWords);
  bool IsTagValid = Ty->isPointerTy();
  SmallVector<APInt::WordType> RawTagBits;
  if (Ty->isPointerTy())
    RawTagBits.resize(NumWords);
  bool IsNoAliasValid = ExperimentalNoAlias && Ty->isPointerTy();
  std::optional<uint64_t> LoadedNoAliasNode;
  bool SawNoAliasBits = false;
  bool SawMissingNoAliasBits = false;
  for (uint32_t I = 0; I < NumBitsToExtract; I += 8) {
    // Try to form a 'logical' byte that represents the bits in the range
    // [BitsStart, BitsEnd].
    uint32_t NumBitsInByte = std::min(8U, NumBitsToExtract - I);
    uint32_t BitsStart = OffsetInBits + I;
    uint32_t BitsEnd = BitsStart + NumBitsInByte - 1;
    Byte LogicalByte;
    // Check whether it is a cross-byte access.
    if (((BitsStart ^ BitsEnd) & ~7) == 0)
      LogicalByte = Bytes[BitsStart / 8].lshr(BitsStart % 8);
    else
      LogicalByte =
          Byte::fshr(Bytes[BitsStart / 8], Bytes[BitsEnd / 8], BitsStart % 8);

    uint32_t Mask = (1U << NumBitsInByte) - 1;
    // If any of the bits in the byte is poison, the whole value is poison.
    if (~LogicalByte.ConcreteMask & ~LogicalByte.Value & Mask) {
      if (ContainsUndefinedBits)
        *ContainsUndefinedBits = true;
      OffsetInBits = NewOffsetInBits;
      return AnyValue::poison();
    }
    uint8_t RandomBits = 0;
    if (~LogicalByte.ConcreteMask & Mask) {
      // This byte contains undef bits.
      if (ContainsUndefinedBits)
        *ContainsUndefinedBits = true;

      if (getEffectiveUndefValueBehavior() ==
          UndefValueBehavior::NonDeterministic) {
        // We don't use std::uniform_int_distribution here because it produces
        // different results across different library implementations. Instead,
        // we directly use the low bits from Rng.
        RandomBits = static_cast<uint8_t>(Rng());
      }
    }
    uint8_t ActualBits = ((LogicalByte.Value & LogicalByte.ConcreteMask) |
                          (RandomBits & ~LogicalByte.ConcreteMask)) &
                         Mask;
    RawBits[I / WordBits] |= static_cast<APInt::WordType>(ActualBits)
                             << (I % WordBits);
    if (IsTagValid) {
      if ((LogicalByte.TagMask & LogicalByte.ConcreteMask & Mask) == Mask) {
        uint8_t ActualTagBits = LogicalByte.TagValue & Mask;
        RawTagBits[I / WordBits] |= static_cast<APInt::WordType>(ActualTagBits)
                                    << (I % WordBits);
      } else {
        IsTagValid = false;
      }
    }
    if (IsNoAliasValid) {
      uint8_t NoAliasMask = LogicalByte.NoAliasMask & Mask;
      if (NoAliasMask == Mask) {
        SawNoAliasBits = true;
        if (!LoadedNoAliasNode)
          LoadedNoAliasNode = LogicalByte.NoAliasNode;
        else if (*LoadedNoAliasNode != LogicalByte.NoAliasNode)
          IsNoAliasValid = false;
      } else if (!NoAliasMask) {
        SawMissingNoAliasBits = true;
      } else {
        IsNoAliasValid = false;
      }
      if (SawNoAliasBits && SawMissingNoAliasBits)
        IsNoAliasValid = false;
    }
  }
  OffsetInBits = NewOffsetInBits;

  APInt Bits(NumBitsToExtract, RawBits);

  // Padding bits for non-byte-sized scalar types must be zero.
  if (NeedsPadding) {
    if (!Bits.isIntN(NumBits)) {
      if (ContainsUndefinedBits)
        *ContainsUndefinedBits = true;
      return AnyValue::poison();
    }
    Bits = Bits.trunc(NumBits);
  }

  if (Ty->isIntegerTy())
    return Bits;
  if (Ty->isFloatingPointTy())
    return APFloat(Ty->getFltSemantics(), Bits);
  assert(Ty->isPointerTy() && "Expect a pointer type");
  // Try to recover provenance from the tag.
  if (IsTagValid) {
    APInt Tag(NumBitsToExtract, RawTagBits);
    if (auto Prov = TaggedProvenances.lookup(Tag))
      return Pointer(std::move(Prov), Bits,
                     IsNoAliasValid && LoadedNoAliasNode ? *LoadedNoAliasNode
                                                         : 0);
  }
  return Pointer(Bits);
}

AnyValue Context::fromBytes(ArrayRef<Byte> Bytes, Type *Ty,
                            bool *ContainsUndefinedBits) {
  assert(Bytes.size() == getEffectiveTypeStoreSize(Ty) &&
         "Invalid byte array size for the type");
  if (Ty->isIntegerTy() || Ty->isFloatingPointTy() || Ty->isPointerTy())
    return fromBytes(ConstBytesView(Bytes, DL), Ty, /*OffsetInBits=*/0,
                     /*CheckPaddingBits=*/true, ContainsUndefinedBits);

  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    Type *ElemTy = VecTy->getElementType();
    uint32_t ElemBits = DL.getTypeSizeInBits(ElemTy).getFixedValue();
    uint32_t NumElements = getEVL(VecTy->getElementCount());
    // Check padding bits. <N x iM> acts as if an integer type with N * M bits.
    uint32_t VecBits = ElemBits * NumElements;
    uint32_t AlignedVecBits = alignTo(VecBits, 8);
    ConstBytesView View(Bytes, DL);
    if (VecBits != AlignedVecBits) {
      const Byte &PaddingByte = View[Bytes.size() - 1];
      uint32_t Mask = (~0U << (VecBits % 8)) & 255U;
      // Make sure all high padding bits are zero.
      if ((PaddingByte.ConcreteMask & ~PaddingByte.Value & Mask) != Mask) {
        if (ContainsUndefinedBits)
          *ContainsUndefinedBits = true;
        return AnyValue::getPoisonValue(*this, Ty);
      }
    }

    std::vector<AnyValue> ValVec;
    ValVec.reserve(NumElements);
    // For little endian element zero is put in the least significant bits of
    // the integer, and for big endian element zero is put in the most
    // significant bits.
    for (uint32_t I = 0; I != NumElements; ++I)
      ValVec.push_back(
          fromBytes(View, ElemTy,
                    DL.isLittleEndian() ? I * ElemBits
                                        : VecBits - ElemBits - I * ElemBits,
                    /*CheckPaddingBits=*/false, ContainsUndefinedBits));
    return AnyValue(std::move(ValVec));
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    Type *ElemTy = ArrTy->getElementType();
    uint64_t Stride = getEffectiveTypeAllocSize(ElemTy);
    uint64_t StoreSize = getEffectiveTypeStoreSize(ElemTy);
    uint32_t NumElements = ArrTy->getNumElements();
    std::vector<AnyValue> ValVec;
    ValVec.reserve(NumElements);
    for (uint32_t I = 0; I != NumElements; ++I)
      ValVec.push_back(fromBytes(Bytes.slice(I * Stride, StoreSize), ElemTy,
                                 ContainsUndefinedBits));
    return AnyValue(std::move(ValVec));
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    const StructLayout *Layout = DL.getStructLayout(StructTy);
    std::vector<AnyValue> ValVec;
    uint32_t NumElements = StructTy->getNumElements();
    ValVec.reserve(NumElements);
    for (uint32_t I = 0; I != NumElements; ++I) {
      Type *ElemTy = StructTy->getElementType(I);
      ValVec.push_back(fromBytes(
          Bytes.slice(getEffectiveTypeSize(Layout->getElementOffset(I)),
                      getEffectiveTypeStoreSize(ElemTy)),
          ElemTy, ContainsUndefinedBits));
    }
    return AnyValue(std::move(ValVec));
  }
  llvm_unreachable("Unsupported first class type.");
}

void Context::toBytes(const AnyValue &Val, Type *Ty, uint32_t OffsetInBits,
                      MutableBytesView Bytes, bool PaddingBits) {
  uint32_t NumBits = DL.getTypeSizeInBits(Ty).getFixedValue();
  uint32_t NewOffsetInBits = OffsetInBits + NumBits;
  if (PaddingBits)
    NewOffsetInBits = alignTo(NewOffsetInBits, 8);
  bool NeedsPadding = NewOffsetInBits != OffsetInBits + NumBits;
  auto WriteBits = [&](const APInt &Bits, const APInt *TagBits,
                       uint64_t NoAliasNode) {
    for (uint32_t I = 0, E = Bits.getBitWidth(); I < E; I += 8) {
      uint32_t NumBitsInByte = std::min(8U, E - I);
      uint32_t BitsStart = OffsetInBits + I;
      uint32_t BitsEnd = BitsStart + NumBitsInByte - 1;
      uint8_t BitsVal =
          static_cast<uint8_t>(Bits.extractBitsAsZExtValue(NumBitsInByte, I));

      Bytes[BitsStart / 8].writeBits(
          static_cast<uint8_t>(((1U << NumBitsInByte) - 1) << (BitsStart % 8)),
          static_cast<uint8_t>(BitsVal << (BitsStart % 8)));
      // If it is a cross-byte access, write the remaining bits to the next
      // byte.
      if (((BitsStart ^ BitsEnd) & ~7) != 0)
        Bytes[BitsEnd / 8].writeBits(
            static_cast<uint8_t>((1U << (BitsEnd % 8 + 1)) - 1),
            static_cast<uint8_t>(BitsVal >> (8 - (BitsStart % 8))));

      if (TagBits) {
        uint8_t TagBitsVal = static_cast<uint8_t>(
            TagBits->extractBitsAsZExtValue(NumBitsInByte, I));
        Bytes[BitsStart / 8].writeTagBits(
            static_cast<uint8_t>(((1U << NumBitsInByte) - 1)
                                 << (BitsStart % 8)),
            static_cast<uint8_t>(TagBitsVal << (BitsStart % 8)));
        // If it is a cross-byte access, write the remaining bits to the next
        // byte.
        if (((BitsStart ^ BitsEnd) & ~7) != 0)
          Bytes[BitsEnd / 8].writeTagBits(
              static_cast<uint8_t>((1U << (BitsEnd % 8 + 1)) - 1),
              static_cast<uint8_t>(TagBitsVal >> (8 - (BitsStart % 8))));
      }
      if (NoAliasNode) {
        Bytes[BitsStart / 8].writeNoAliasBits(
            static_cast<uint8_t>(((1U << NumBitsInByte) - 1)
                                 << (BitsStart % 8)),
            NoAliasNode);
        if (((BitsStart ^ BitsEnd) & ~7) != 0)
          Bytes[BitsEnd / 8].writeNoAliasBits(
              static_cast<uint8_t>((1U << (BitsEnd % 8 + 1)) - 1), NoAliasNode);
      }
    }
  };
  if (Val.isPoison()) {
    for (uint32_t I = 0, E = NewOffsetInBits - OffsetInBits; I < E;) {
      uint32_t NumBitsInByte = std::min(8 - (OffsetInBits + I) % 8, E - I);
      assert(((OffsetInBits ^ (OffsetInBits + NumBitsInByte - 1)) & ~7) == 0 &&
             "Across byte boundary.");
      Bytes[(OffsetInBits + I) / 8].poisonBits(static_cast<uint8_t>(
          ((1U << NumBitsInByte) - 1) << ((OffsetInBits + I) % 8)));
      I += NumBitsInByte;
    }
  } else if (Ty->isIntegerTy()) {
    auto &Bits = Val.asInteger();
    WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits) : Bits,
              /*TagBits=*/nullptr, /*NoAliasNode=*/0);
  } else if (Ty->isFloatingPointTy()) {
    auto Bits = Val.asFloat().bitcastToAPInt();
    WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits) : Bits,
              /*TagBits=*/nullptr, /*NoAliasNode=*/0);
  } else if (Ty->isPointerTy()) {
    auto &AddressBits = Val.asPointer().address();
    APInt Tag = getTag(AddressBits.getBitWidth(), Val.asPointer().provenance());
    if (NeedsPadding)
      Tag = Tag.zext(NewOffsetInBits - OffsetInBits);
    uint64_t NoAliasNode =
        ExperimentalNoAlias ? Val.asPointer().getNoAliasNodeID() : 0;
    WriteBits(NeedsPadding ? AddressBits.zext(NewOffsetInBits - OffsetInBits)
                           : AddressBits,
              &Tag, NoAliasNode);
  } else {
    llvm_unreachable("Unsupported scalar type.");
  }
}

void Context::toBytes(const AnyValue &Val, Type *Ty,
                      MutableArrayRef<Byte> Bytes) {
  assert(Bytes.size() == getEffectiveTypeStoreSize(Ty) &&
         "Invalid byte array size for the type");
  if (Ty->isIntegerTy() || Ty->isFloatingPointTy() || Ty->isPointerTy()) {
    toBytes(Val, Ty, /*OffsetInBits=*/0, MutableBytesView(Bytes, DL),
            /*PaddingBits=*/true);
    return;
  }

  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    Type *ElemTy = VecTy->getElementType();
    uint32_t ElemBits = DL.getTypeSizeInBits(ElemTy).getFixedValue();
    uint32_t NumElements = getEVL(VecTy->getElementCount());
    // Zero padding bits. <N x iM> acts as if an integer type with N * M bits.
    uint32_t VecBits = ElemBits * NumElements;
    uint32_t AlignedVecBits = alignTo(VecBits, 8);
    MutableBytesView View(Bytes, DL);
    if (VecBits != AlignedVecBits) {
      Byte &PaddingByte = View[Bytes.size() - 1];
      uint32_t Mask = (~0U << (VecBits % 8)) & 255U;
      PaddingByte.zeroBits(Mask);
    }
    // For little endian element zero is put in the least significant bits of
    // the integer, and for big endian element zero is put in the most
    // significant bits.
    if (DL.isLittleEndian()) {
      for (const auto &[I, Val] : enumerate(Val.asAggregate()))
        toBytes(Val, ElemTy, ElemBits * I, View, /*PaddingBits=*/false);
    } else {
      for (const auto &[I, Val] : enumerate(reverse(Val.asAggregate())))
        toBytes(Val, ElemTy, ElemBits * I, View, /*PaddingBits=*/false);
    }
    return;
  }

  // Fill padding bytes due to alignment requirement.
  auto FillUndefBytes = [&](uint64_t Begin, uint64_t End) {
    fill(Bytes.slice(Begin, End - Begin), Byte::undef());
  };
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    Type *ElemTy = ArrTy->getElementType();
    uint64_t Offset = 0;
    uint64_t Stride = getEffectiveTypeAllocSize(ElemTy);
    uint64_t StoreSize = getEffectiveTypeStoreSize(ElemTy);
    for (const auto &SubVal : Val.asAggregate()) {
      toBytes(SubVal, ElemTy, Bytes.slice(Offset, StoreSize));
      FillUndefBytes(Offset + StoreSize, Offset + Stride);
      Offset += Stride;
    }
    return;
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    const StructLayout *Layout = DL.getStructLayout(StructTy);
    uint64_t LastAccessedOffset = 0;
    for (uint32_t I = 0, E = Val.asAggregate().size(); I != E; ++I) {
      Type *ElemTy = StructTy->getElementType(I);
      uint64_t ElemOffset = getEffectiveTypeSize(Layout->getElementOffset(I));
      uint64_t ElemStoreSize = getEffectiveTypeStoreSize(ElemTy);
      FillUndefBytes(LastAccessedOffset, ElemOffset);
      toBytes(Val.asAggregate()[I], ElemTy,
              Bytes.slice(ElemOffset, ElemStoreSize));
      LastAccessedOffset = ElemOffset + ElemStoreSize;
    }
    FillUndefBytes(LastAccessedOffset, getEffectiveTypeStoreSize(StructTy));
    return;
  }

  llvm_unreachable("Unsupported first class type.");
}

AnyValue Context::load(MemoryObject &MO, uint64_t Offset, Type *ValTy,
                       bool *ContainsUndefinedBits) {
  return fromBytes(
      MO.getBytes().slice(Offset, getEffectiveTypeStoreSize(ValTy)), ValTy,
      ContainsUndefinedBits);
}

void Context::store(MemoryObject &MO, uint64_t Offset, const AnyValue &Val,
                    Type *ValTy) {
  toBytes(Val, ValTy,
          MO.getBytes().slice(Offset, getEffectiveTypeStoreSize(ValTy)));
}

void Context::storeRawBytes(MemoryObject &MO, uint64_t Offset, const void *Data,
                            uint64_t Size) {
  for (uint64_t I = 0; I != Size; ++I)
    MO[Offset + I] = Byte::concrete(static_cast<const uint8_t *>(Data)[I]);
}

APInt Context::generateRandomAPInt(uint32_t BitWidth) {
  SmallVector<APInt::WordType> RandomWords;
  uint32_t NumWords = APInt::getNumWords(BitWidth);
  RandomWords.reserve(NumWords);
  static_assert(decltype(Rng)::word_size >=
                    std::numeric_limits<APInt::WordType>::digits,
                "Unexpected Rng result type.");
  for (uint32_t I = 0; I != NumWords; ++I)
    RandomWords.push_back(static_cast<APInt::WordType>(Rng()));
  return APInt(BitWidth, RandomWords);
}

void Context::freeze(AnyValue &Val, Type *Ty) {
  if (Val.isPoison()) {
    uint32_t Bits = DL.getTypeSizeInBits(Ty);
    APInt RandomVal = mayUseNonDeterminism() ? generateRandomAPInt(Bits)
                                             : APInt::getZero(Bits);
    if (Ty->isIntegerTy())
      Val = AnyValue(RandomVal);
    else if (Ty->isFloatingPointTy())
      Val = AnyValue(APFloat(Ty->getFltSemantics(), RandomVal));
    else if (Ty->isPointerTy())
      Val = AnyValue(Pointer(RandomVal));
    else
      llvm_unreachable("Unsupported scalar type for poison value");
    return;
  }
  if (Val.isAggregate()) {
    auto &SubVals = Val.asAggregate();
    if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
      Type *ElemTy = VecTy->getElementType();
      for (auto &SubVal : SubVals)
        freeze(SubVal, ElemTy);
    } else if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
      Type *ElemTy = ArrTy->getElementType();
      for (auto &SubVal : SubVals)
        freeze(SubVal, ElemTy);
    } else if (auto *StructTy = dyn_cast<StructType>(Ty)) {
      for (uint32_t I = 0, E = SubVals.size(); I != E; ++I)
        freeze(SubVals[I], StructTy->getElementType(I));
    } else {
      llvm_unreachable("Invalid aggregate type");
    }
  }
}

AnyValue Context::computePtrAdd(const Pointer &Ptr, const APInt &Offset,
                                GEPNoWrapFlags Flags,
                                AnyValue &AccumulatedOffset) {
  if (Offset.isZero())
    return Ptr;
  APInt IndexBits = Ptr.address().trunc(Offset.getBitWidth());
  auto NewIndex =
      addNoWrap(IndexBits, Offset, /*HasNSW=*/false, Flags.hasNoUnsignedWrap());
  if (NewIndex.isPoison())
    return AnyValue::poison();
  if (Flags.hasNoUnsignedSignedWrap()) {
    // The successive addition of the current address, truncated to the
    // pointer index type and interpreted as an unsigned number, and each
    // offset, interpreted as a signed number, does not wrap the pointer index
    // type.
    if (Offset.isNonNegative() ? NewIndex.asInteger().ult(IndexBits)
                               : NewIndex.asInteger().ugt(IndexBits))
      return AnyValue::poison();
  }
  APInt NewAddr = Ptr.address();
  NewAddr.insertBits(NewIndex.asInteger(), 0);

  MemoryObject *MO = nullptr;
  if (Flags.isInBounds()) {
    MO = checkProvenance(
        Ptr, [](const Provenance &) { return true; },
        /*HasSideEffect=*/false);
    if (!MO || !MO->inBounds(NewAddr))
      return AnyValue::poison();
  }

  if (!AccumulatedOffset.isPoison()) {
    AccumulatedOffset =
        addNoWrap(AccumulatedOffset.asInteger(), Offset,
                  Flags.hasNoUnsignedSignedWrap(), Flags.hasNoUnsignedWrap());
    if (AccumulatedOffset.isPoison())
      return AnyValue::poison();
  }

  // Should not expose provenance here even if the new address doesn't point
  // to the original object.
  auto Res = Ptr.getWithNewAddr(NewAddr);
  if (MO) {
    auto &Prov = Res.provenance();
    if (Prov.isWildcard() && !Prov.getMemoryObject())
      Res = Res.getWithNewProvenance(Prov.getWithKnownMemoryObject(*MO));
  }
  return Res;
}

AnyValue Context::computePtrAdd(const AnyValue &Ptr, const APInt &Offset,
                                GEPNoWrapFlags Flags,
                                AnyValue &AccumulatedOffset) {
  if (Ptr.isPoison())
    return AnyValue::poison();
  return computePtrAdd(Ptr.asPointer(), Offset, Flags, AccumulatedOffset);
}

AnyValue Context::computeScaledPtrAdd(const AnyValue &Ptr,
                                      const AnyValue &Index, const APInt &Scale,
                                      GEPNoWrapFlags Flags,
                                      AnyValue &AccumulatedOffset) {
  if (Ptr.isPoison() || Index.isPoison())
    return AnyValue::poison();
  assert(Ptr.isPointer() && Index.isInteger() && "Unexpected type.");
  if (Scale.isOne())
    return computePtrAdd(Ptr, Index.asInteger(), Flags, AccumulatedOffset);
  auto ScaledOffset =
      mulNoWrap(Index.asInteger(), Scale, Flags.hasNoUnsignedSignedWrap(),
                Flags.hasNoUnsignedWrap());
  if (ScaledOffset.isPoison())
    return AnyValue::poison();
  return computePtrAdd(Ptr, ScaledOffset.asInteger(), Flags, AccumulatedOffset);
}

static AnyValue canonicalizeIndex(const AnyValue &Idx, unsigned IndexBitWidth,
                                  GEPNoWrapFlags Flags) {
  if (Idx.isPoison())
    return AnyValue::poison();
  auto &IdxInt = Idx.asInteger();
  if (IdxInt.getBitWidth() == IndexBitWidth)
    return Idx;
  if (IdxInt.getBitWidth() > IndexBitWidth) {
    if (Flags.hasNoUnsignedSignedWrap() && !IdxInt.isSignedIntN(IndexBitWidth))
      return AnyValue::poison();

    if (Flags.hasNoUnsignedWrap() && !IdxInt.isIntN(IndexBitWidth))
      return AnyValue::poison();

    return IdxInt.trunc(IndexBitWidth);
  }
  return IdxInt.sext(IndexBitWidth);
}

AnyValue
Context::computeGEP(GEPOperator &GEP,
                    function_ref<const AnyValue &(Value *V)> GetValue) {
  uint32_t IndexBitWidth =
      DL.getIndexSizeInBits(GEP.getType()->getPointerAddressSpace());
  GEPNoWrapFlags Flags = GEP.getNoWrapFlags();
  AnyValue Res = GetValue(GEP.getPointerOperand());
  AnyValue AccumulatedOffset = APInt(IndexBitWidth, 0);
  if (Res.isAggregate())
    AccumulatedOffset =
        AnyValue::getVectorSplat(AccumulatedOffset, Res.asAggregate().size());
  auto ApplyScaledOffset = [&](const AnyValue &Index, const APInt &Scale) {
    if (Index.isAggregate() && !Res.isAggregate()) {
      Res = AnyValue::getVectorSplat(Res, Index.asAggregate().size());
      AccumulatedOffset = AnyValue::getVectorSplat(AccumulatedOffset,
                                                   Index.asAggregate().size());
    }
    if (Index.isAggregate() && Res.isAggregate()) {
      for (auto &&[ResElem, IndexElem, OffsetElem] :
           zip(Res.asAggregate(), Index.asAggregate(),
               AccumulatedOffset.asAggregate()))
        ResElem = computeScaledPtrAdd(
            ResElem, canonicalizeIndex(IndexElem, IndexBitWidth, Flags), Scale,
            Flags, OffsetElem);
    } else {
      AnyValue CanonicalIndex = canonicalizeIndex(Index, IndexBitWidth, Flags);
      if (Res.isAggregate()) {
        for (auto &&[ResElem, OffsetElem] :
             zip(Res.asAggregate(), AccumulatedOffset.asAggregate()))
          ResElem = computeScaledPtrAdd(ResElem, CanonicalIndex, Scale, Flags,
                                        OffsetElem);
      } else {
        Res = computeScaledPtrAdd(Res, CanonicalIndex, Scale, Flags,
                                  AccumulatedOffset);
      }
    }
  };

  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    Value *V = GTI.getOperand();

    // Fast path for zero offsets.
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      if (CI->isZero())
        continue;
    }
    if (isa<ConstantAggregateZero>(V))
      continue;

    // Handle a struct index, which adds its field offset to the pointer.
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      unsigned ElementIdx = cast<ConstantInt>(V)->getZExtValue();
      const StructLayout *SL = DL.getStructLayout(STy);
      // Element offset is in bytes.
      ApplyScaledOffset(APInt(IndexBitWidth, SL->getElementOffset(ElementIdx)),
                        APInt(IndexBitWidth, 1));
      continue;
    }

    // Truncate if type size exceeds index space.
    // TODO: Should be documented in LangRef: GEPs with nowrap flags should
    // return poison when the type size exceeds index space.
    TypeSize Offset = GTI.getSequentialElementStride(DL);
    APInt Scale(IndexBitWidth, getEffectiveTypeSize(Offset),
                /*isSigned=*/false, /*implicitTrunc=*/true);
    if (!Scale.isZero())
      ApplyScaledOffset(GetValue(V), Scale);
  }
  return Res;
}

MemoryObject::~MemoryObject() = default;
MemoryObject::MemoryObject(uint64_t Addr, uint64_t Size, StringRef Name,
                           unsigned AS, MemInitKind InitKind,
                           MemAllocKind AllocKind, bool IsIRGlobalValue)
    : Address(Addr), Size(Size), Name(Name), AS(AS),
      State(InitKind != MemInitKind::Poisoned ? MemoryObjectState::Alive
                                              : MemoryObjectState::Dead),
      AllocKind(AllocKind), IsIRGlobalValue(IsIRGlobalValue) {
  switch (InitKind) {
  case MemInitKind::Zeroed:
    Bytes.resize(Size, Byte::concrete(0));
    break;
  case MemInitKind::Uninitialized:
    Bytes.resize(Size, Byte::undef());
    break;
  case MemInitKind::Poisoned:
    Bytes.resize(Size, Byte::poison());
    break;
  }
}

IntrusiveRefCntPtr<MemoryObject>
Context::allocate(uint64_t Size, uint64_t Align, StringRef Name, unsigned AS,
                  MemInitKind InitKind, MemAllocKind AllocKind,
                  bool IsIRGlobalValue) {
  // Even if the memory object is zero-sized, it still occupies a byte to obtain
  // a unique address.
  uint64_t AllocateSize = std::max(Size, (uint64_t)1);
  if (MaxMem != 0 && SaturatingAdd(UsedMem, AllocateSize) >= MaxMem)
    return nullptr;
  uint64_t AlignedAddr = alignTo(AllocationBase, Align);
  auto MemObj = makeIntrusiveRefCnt<MemoryObject>(
      AlignedAddr, Size, Name, AS, InitKind, AllocKind, IsIRGlobalValue);
  MemoryObjects[AlignedAddr] = MemObj;
  // Extra padding to make sure getWildcardProvenance resolves to at most one
  // memory object.
  AllocationBase = AlignedAddr + AllocateSize + 1;
  UsedMem += AllocateSize;
  return MemObj;
}

bool Context::free(const MemoryObject &Obj) {
  uint64_t Address = Obj.getAddress();
  auto It = MemoryObjects.find(Address);
  if (It == MemoryObjects.end() || It->second.get() != &Obj)
    return false;

  UsedMem -= std::max(It->second->getSize(), static_cast<uint64_t>(1));

  clearNoAliasState(*It->second);
  MemoryObject &MutableObj = *It->second;
  MutableObj.State = MemoryObjectState::Freed;
  MutableObj.Bytes.clear();
  for (const APInt &Tag : MutableObj.AssociatedTags)
    TaggedProvenances.erase(Tag);
  MutableObj.AssociatedTags.clear();
  ExposedProvenances.erase(Address);

  MemoryObjects.erase(It);
  return true;
}

Pointer Context::deriveFromMemoryObject(IntrusiveRefCntPtr<MemoryObject> Obj) {
  assert(Obj && "Cannot determine the address space of a null memory object");
  return Pointer(makeIntrusiveRefCnt<Provenance>(Obj),
                 APInt(DL.getPointerSizeInBits(Obj->getAddressSpace()),
                       Obj->getAddress()));
}

void Context::exposeProvenance(Provenance &Prov) {
  if (Prov.Wildcard)
    return;
  MemoryObject *Obj = Prov.getMemoryObject();
  if (!Obj)
    return;
  uint64_t Address = Obj->getAddress();
  ExposedProvenanceSet &Set = ExposedProvenances[Address];
  if (Set.Set.insert(&Prov).second)
    Set.List.push_back({&Prov, ++ExposedProvenanceSetGeneration});
}

MemoryObject *
Context::checkProvenance(const Pointer &Ptr,
                         function_ref<bool(const Provenance &)> Check,
                         bool HasSideEffect) {
  auto &Prov = Ptr.provenance();
  if (!Check(Prov))
    return nullptr;
  // Early return for concrete provenances.
  if (!Prov.Wildcard)
    return Prov.Obj.get();

  MemoryObject *MO = nullptr;
  APInt &Mask = Prov.Wildcard->ActiveMask;
  SmallVector<ExposedProvenance> *List = nullptr;
  uint32_t ProvenanceCount = 0;
  if (Mask.isZero()) {
    // The memory object hasn't been determined.
    uint64_t Addr = Ptr.address().getLimitedValue();
    auto Iter = ExposedProvenances.upper_bound(Addr);
    if (Iter == ExposedProvenances.begin())
      return nullptr;
    auto &[BaseAddress, Set] = *std::prev(Iter);
    auto &Obj = MemoryObjects.at(BaseAddress);
    if (!Obj->inBounds(Ptr.address()))
      return nullptr;
    MO = Obj.get();
    // We only inspect the first N exposed provenances according to the global
    // generation number of the wildcard pointer.
    ProvenanceCount = std::distance(
        Set.List.begin(),
        upper_bound(Set.List,
                    ExposedProvenance{nullptr, Prov.Wildcard->Generation}));
    if (HasSideEffect) {
      Mask = APInt::getAllOnes(ProvenanceCount);
      Prov.Wildcard->BaseAddress = BaseAddress;
    }
    List = &Set.List;
  } else {
    // We already determined the memory object in a previous memory access.
    uint64_t BaseAddress = Prov.Wildcard->BaseAddress;
    auto Iter = ExposedProvenances.find(BaseAddress);
    // The memory object has been freed.
    if (Iter == ExposedProvenances.end())
      return nullptr;
    MO = MemoryObjects.at(BaseAddress).get();
    if (!MO->inBounds(Ptr.address()))
      return nullptr;
    List = &Iter->second.List;
    ProvenanceCount = Mask.getBitWidth();
  }
  if (Prov.Obj) {
    // We already determined the memory object via speculatable operations like
    // gep inbounds.
    if (Prov.Obj.get() != MO)
      return nullptr;
  }

  bool Valid = false;
  for (uint32_t I = 0; I != ProvenanceCount; ++I) {
    assert((!HasSideEffect || !Mask.isZero()) &&
           "Mask must be initialized if HasSideEffect is true.");
    if (!Mask.isZero() && !Mask[I])
      continue;
    if (Check(*(*List)[I].Prov)) {
      Valid = true;
      // Early return as we don't need to update the Mask.
      if (!HasSideEffect)
        break;
    } else if (HasSideEffect)
      Mask.clearBit(I);
  }

  return Valid ? MO : nullptr;
}

IntrusiveRefCntPtr<Provenance> Context::getWildcardProvenance() {
  // No exposed provenances.
  if (ExposedProvenanceSetGeneration == 0)
    return Provenance::nullary();
  auto Prov = makeIntrusiveRefCnt<Provenance>(nullptr);
  Prov->Wildcard =
      makeIntrusiveRefCnt<WildcardProvenance>(ExposedProvenanceSetGeneration);
  return Prov;
}

Function *Context::getTargetFunction(const Pointer &Ptr) {
  if (Ptr.address().getActiveBits() > 64)
    return nullptr;
  auto It = ValidFuncTargets.find(Ptr.address().getZExtValue());
  if (It == ValidFuncTargets.end())
    return nullptr;
  // TODO: check the provenance of pointer.
  return It->second.first;
}
BasicBlock *Context::getTargetBlock(const Pointer &Ptr) {
  if (Ptr.address().getActiveBits() > 64)
    return nullptr;
  auto It = ValidBlockTargets.find(Ptr.address().getZExtValue());
  if (It == ValidBlockTargets.end())
    return nullptr;
  // TODO: check the provenance of pointer.
  return It->second.first;
}

uint64_t Context::getEffectiveTypeAllocSize(Type *Ty) {
  // FIXME: It is incorrect for overaligned scalable vector types.
  return getEffectiveTypeSize(DL.getTypeAllocSize(Ty));
}
uint64_t Context::getEffectiveTypeStoreSize(Type *Ty) {
  return getEffectiveTypeSize(DL.getTypeStoreSize(Ty));
}

RoundingMode Context::getCurrentRoundingMode() const {
  return CurrentRoundingMode;
}

fp::ExceptionBehavior Context::getCurrentExceptionBehavior() const {
  return CurrentExceptionBehavior;
}

void Context::setCurrentRoundingMode(RoundingMode RM) {
  CurrentRoundingMode = RM;
}

void Context::setCurrentExceptionBehavior(fp::ExceptionBehavior EB) {
  CurrentExceptionBehavior = EB;
}

bool Context::isDefaultFPEnv() const {
  return isDefaultFPEnvironment(CurrentExceptionBehavior, CurrentRoundingMode);
}

UndefValueBehavior Context::getEffectiveUndefValueBehavior() const {
  if (isDeterministic())
    return UndefValueBehavior::Zero;
  return UndefBehavior;
}

NaNPropagationBehavior Context::getEffectiveNaNPropagationBehavior() const {
  if (isDeterministic())
    return NaNPropagationBehavior::PreferredNaN;
  return NaNBehavior;
}

bool Context::getRandomBool() {
  // We use the lowest bit of the raw bits from RNG as the result:
  if (mayUseNonDeterminism())
    return static_cast<bool>(Rng() & 1);
  return false;
}

uint64_t Context::getRandomUInt64() {
  if (mayUseNonDeterminism())
    return Rng();
  return 0;
}

bool MemoryObject::isGlobal() const {
  return AllocKind == MemAllocKind::Global;
}

bool MemoryObject::isStackAllocated() const {
  return AllocKind == MemAllocKind::Stack;
}

bool MemoryObject::isHeapAllocated() const {
  switch (AllocKind) {
  case MemAllocKind::Global:
  case MemAllocKind::BlockAddress:
  case MemAllocKind::Stack:
    return false;
  case MemAllocKind::Malloc:
  case MemAllocKind::New:
  case MemAllocKind::NewArray:
    return true;
  }

  llvm_unreachable("Unknown MemAllocKind");
}

bool Context::isNoAliasAncestor(uint64_t Ancestor, uint64_t Descendant) const {
  if (!Ancestor || !Descendant)
    return false;
  // Parent links are stable while a descendant is active. If a stale node id
  // was pruned, reaching a missing node means the relationship no longer
  // exists.
  for (uint64_t NodeID = Descendant; NodeID;) {
    if (NodeID == Ancestor)
      return true;
    const auto It = NoAliasNodes.find(NodeID);
    if (It == NoAliasNodes.end())
      return false;
    NodeID = It->second.Parent;
  }
  return false;
}

bool Context::hasActiveNoAliasDescendant(uint64_t NodeID) const {
  for (const auto &[CandidateID, Candidate] : NoAliasNodes) {
    if (!Candidate.Active || CandidateID == NodeID)
      continue;
    if (isNoAliasAncestor(NodeID, CandidateID))
      return true;
  }
  return false;
}

void Context::tryEraseInactiveNoAliasNode(uint64_t NodeID) {
  const auto It = NoAliasNodes.find(NodeID);
  if (It == NoAliasNodes.end() || It->second.Active)
    return;
  if (hasActiveNoAliasDescendant(NodeID))
    return;

  // An inactive node can still be relevant as the parent of a live child. Once
  // that is no longer true, stale pointers carrying this ID should behave like
  // raw/root pointers during future retagging.
  const uint64_t Parent = It->second.Parent;
  appendNoAliasEvent("erased inactive protector " + getNoAliasNodeName(NodeID));
  NoAliasNodes.erase(It);
  if (Parent)
    tryEraseInactiveNoAliasNode(Parent);
}

StringRef Context::getNoAliasAccessKindName(NoAliasAccessKind Kind) {
  switch (Kind) {
  case NoAliasAccessKind::Read:
    return "read";
  case NoAliasAccessKind::Write:
    return "write";
  }
  llvm_unreachable("Unknown NoAliasAccessKind");
}

std::string Context::getNoAliasNodeName(uint64_t NodeID) {
  if (!NodeID)
    return "raw/root";
  std::string S;
  raw_string_ostream OS(S);
  OS << "node #" << NodeID;
  return S;
}

std::string Context::getNoAliasActivationName(uint64_t ActivationID) {
  std::string S;
  raw_string_ostream OS(S);
  OS << "activation #" << ActivationID;
  return S;
}

std::string Context::getNoAliasObjectName(const MemoryObject &MO) {
  if (MO.getName().empty()) {
    std::string S;
    raw_string_ostream OS(S);
    OS << "object at 0x";
    OS.write_hex(MO.getAddress());
    return S;
  }
  return ("'" + MO.getName() + "'").str();
}

void Context::appendNoAliasEvent(std::string Msg) {
  NoAliasEvents.push_back(std::move(Msg));
}

uint64_t Context::classifyNoAliasAccess(const NoAliasActivation &Activation,
                                        const MemoryObject &MO,
                                        uint64_t AccessNode) const {
  for (uint64_t NodeID : Activation.Nodes) {
    const auto It = NoAliasNodes.find(NodeID);
    if (It == NoAliasNodes.end() || !It->second.Active ||
        It->second.Object != &MO)
      continue;
    if (isNoAliasAncestor(NodeID, AccessNode))
      return NodeID;
  }
  return 0;
}

bool Context::updateNoAliasAccesses(NoAliasActivation &Activation,
                                    MemoryObject &MO, uint64_t Begin,
                                    uint64_t End, uint64_t AccessClass,
                                    NoAliasAccessKind Kind,
                                    uint64_t ActivationID) {
  assert(Begin < End && "empty accesses should not reach noalias tracking");

  SmallVector<NoAliasAccessRun, 4> NewRuns;
  auto AppendRun = [&](uint64_t RunBegin, uint64_t RunEnd,
                       NoAliasAccessSummary Summary) {
    if (RunBegin == RunEnd)
      return;
    if (!NewRuns.empty() && NewRuns.back().End == RunBegin &&
        NewRuns.back().Summary == Summary) {
      NewRuns.back().End = RunEnd;
      return;
    }
    NewRuns.push_back({RunBegin, RunEnd, Summary});
  };

  auto DescribeSummary = [&](raw_ostream &OS,
                             const NoAliasAccessSummary &Summary) {
    if (Summary.MultipleClasses) {
      OS << "reads by multiple access classes";
      return;
    }
    OS << (Summary.HasWrite ? "access including a write by " : "reads by ")
       << getNoAliasNodeName(Summary.AccessClass);
  };

  auto AppendTransitioned = [&](uint64_t RunBegin, uint64_t RunEnd,
                                const NoAliasAccessSummary *Old) -> bool {
    if (RunBegin == RunEnd)
      return true;

    const bool IsWrite = Kind == NoAliasAccessKind::Write;
    NoAliasAccessSummary New{AccessClass, false, IsWrite};
    if (Old) {
      New = *Old;
      if (Old->MultipleClasses) {
        if (IsWrite)
          New.HasWrite = true;
      } else if (Old->AccessClass == AccessClass) {
        New.HasWrite |= IsWrite;
      } else if (Old->HasWrite || IsWrite) {
        New.MultipleClasses = true;
        New.HasWrite = true;
      } else {
        New.AccessClass = 0;
        New.MultipleClasses = true;
      }
    }

    if (New.MultipleClasses && New.HasWrite) {
      std::string S;
      raw_string_ostream OS(S);
      OS << "noalias violation: " << getNoAliasAccessKindName(Kind)
         << " through " << getNoAliasNodeName(AccessClass) << " on "
         << getNoAliasObjectName(MO) << " bytes [" << RunBegin << ", " << RunEnd
         << ") combines multiple access classes with a write in "
         << getNoAliasActivationName(ActivationID);
      LastNoAliasError = std::move(S);
      appendNoAliasEvent(LastNoAliasError);
      return false;
    }

    std::string S;
    raw_string_ostream OS(S);
    OS << getNoAliasActivationName(ActivationID) << ' '
       << getNoAliasAccessKindName(Kind) << " through "
       << getNoAliasNodeName(AccessClass) << " on " << getNoAliasObjectName(MO)
       << " bytes [" << RunBegin << ", " << RunEnd << "): ";
    if (Old)
      DescribeSummary(OS, *Old);
    else
      OS << "unaccessed";
    OS << " -> ";
    DescribeSummary(OS, New);
    appendNoAliasEvent(std::move(S));
    AppendRun(RunBegin, RunEnd, New);
    return true;
  };

  auto &Runs = Activation.Accesses[&MO];
  uint64_t Cur = Begin;
  bool InsertedAccessTail = false;
  for (const NoAliasAccessRun &Run : Runs) {
    if (Run.End <= Begin) {
      AppendRun(Run.Begin, Run.End, Run.Summary);
      continue;
    }
    if (Run.Begin >= End) {
      if (!InsertedAccessTail) {
        if (!AppendTransitioned(Cur, End, nullptr))
          return false;
        InsertedAccessTail = true;
      }
      AppendRun(Run.Begin, Run.End, Run.Summary);
      continue;
    }

    if (Run.Begin < Begin)
      AppendRun(Run.Begin, Begin, Run.Summary);

    const uint64_t OverlapBegin = std::max(Cur, Run.Begin);
    if (!AppendTransitioned(Cur, OverlapBegin, nullptr))
      return false;

    const uint64_t OverlapEnd = std::min(End, Run.End);
    if (!AppendTransitioned(OverlapBegin, OverlapEnd, &Run.Summary))
      return false;
    Cur = OverlapEnd;

    if (Run.End > End) {
      AppendRun(End, Run.End, Run.Summary);
      InsertedAccessTail = true;
    }
  }
  if (!InsertedAccessTail) {
    if (!AppendTransitioned(Cur, End, nullptr))
      return false;
  }

  Runs = std::move(NewRuns);
  return true;
}

uint64_t Context::beginNoAliasActivation() {
  if (!ExperimentalNoAlias)
    return 0;
  const uint64_t ActivationID = NextNoAliasActivation++;
  NoAliasActivations.try_emplace(ActivationID);
  return ActivationID;
}

Pointer Context::createNoAliasPointer(const Pointer &Ptr,
                                      uint64_t ActivationID) {
  if (!ExperimentalNoAlias || !ActivationID)
    return Ptr;

  MemoryObject *MO = Ptr.getMemoryObject();
  if (!MO)
    return Ptr;

  auto ActivationIt = NoAliasActivations.find(ActivationID);
  assert(ActivationIt != NoAliasActivations.end() &&
         "Noalias activation must be live before creating a parameter node.");

  const uint64_t NodeID = NextNoAliasNode++;
  uint64_t Parent = Ptr.getNoAliasNodeID();
  // If the parent node was pruned after its activation ended, the incoming
  // pointer is treated as a raw/root-derived pointer for this new activation.
  if (Parent && NoAliasNodes.find(Parent) == NoAliasNodes.end())
    Parent = 0;
  NoAliasNode Node;
  Node.Parent = Parent;
  Node.Object = MO;
  Node.Active = true;
  NoAliasNodes.try_emplace(NodeID, std::move(Node));
  ActivationIt->second.Nodes.push_back(NodeID);

  auto &Activations = NoAliasActivationsByObject[MO];
  if (std::find(Activations.begin(), Activations.end(), ActivationID) ==
      Activations.end())
    Activations.push_back(ActivationID);

  std::string S;
  raw_string_ostream OS(S);
  OS << "created protector " << getNoAliasNodeName(NodeID) << " for "
     << getNoAliasObjectName(*MO) << " in "
     << getNoAliasActivationName(ActivationID) << " based on "
     << getNoAliasNodeName(Parent);
  appendNoAliasEvent(std::move(S));
  return Ptr.getWithNoAliasNode(NodeID);
}

bool Context::accessNoAlias(MemoryObject &MO, uint64_t Offset, uint64_t Size,
                            uint64_t AccessNode, NoAliasAccessKind Kind) {
  if (!ExperimentalNoAlias || !Size)
    return true;

  auto It = NoAliasActivationsByObject.find(&MO);
  if (It == NoAliasActivationsByObject.end())
    return true;

  const uint64_t End = Offset + Size;
  uint32_t CheckedActivations = 0;
  for (uint64_t ActivationID : It->second) {
    auto ActivationIt = NoAliasActivations.find(ActivationID);
    if (ActivationIt == NoAliasActivations.end())
      continue;
    ++CheckedActivations;
    const uint64_t AccessClass =
        classifyNoAliasAccess(ActivationIt->second, MO, AccessNode);
    if (!updateNoAliasAccesses(ActivationIt->second, MO, Offset, End,
                               AccessClass, Kind, ActivationID))
      return false;
  }

  if (CheckedActivations) {
    std::string S;
    raw_string_ostream OS(S);
    OS << getNoAliasAccessKindName(Kind) << " through "
       << getNoAliasNodeName(AccessNode) << " on " << getNoAliasObjectName(MO)
       << " bytes [" << Offset << ", " << End << ") checked "
       << CheckedActivations << " active noalias activation"
       << (CheckedActivations == 1 ? "" : "s");
    appendNoAliasEvent(std::move(S));
  }
  return true;
}

void Context::endNoAliasActivation(uint64_t ActivationID) {
  if (!ExperimentalNoAlias)
    return;

  auto ActivationIt = NoAliasActivations.find(ActivationID);
  if (ActivationIt == NoAliasActivations.end())
    return;

  SmallVector<uint64_t, 4> Nodes(ActivationIt->second.Nodes.begin(),
                                 ActivationIt->second.Nodes.end());
  for (uint64_t NodeID : Nodes) {
    auto NodeIt = NoAliasNodes.find(NodeID);
    if (NodeIt == NoAliasNodes.end())
      continue;
    MemoryObject *MO = NodeIt->second.Object;
    NodeIt->second.Active = false;
    tryEraseInactiveNoAliasNode(NodeID);
    auto ObjectIt = NoAliasActivationsByObject.find(MO);
    if (ObjectIt == NoAliasActivationsByObject.end())
      continue;
    auto &IDs = ObjectIt->second;
    IDs.erase(std::remove(IDs.begin(), IDs.end(), ActivationID), IDs.end());
    if (IDs.empty())
      NoAliasActivationsByObject.erase(ObjectIt);
  }
  appendNoAliasEvent("ended " + getNoAliasActivationName(ActivationID));
  NoAliasActivations.erase(ActivationIt);
}

SmallVector<std::string, 4> Context::takeNoAliasEvents() {
  SmallVector<std::string, 8> Events;
  Events.swap(NoAliasEvents);
  return Events;
}

void Context::clearNoAliasState(const MemoryObject &MO) {
  if (!ExperimentalNoAlias)
    return;

  const auto It = NoAliasActivationsByObject.find(&MO);
  if (It == NoAliasActivationsByObject.end())
    return;
  SmallVector<uint64_t, 4> ActivationIDs(It->second.begin(), It->second.end());
  for (uint64_t ActivationID : ActivationIDs) {
    auto ActivationIt = NoAliasActivations.find(ActivationID);
    if (ActivationIt == NoAliasActivations.end())
      continue;
    for (uint64_t NodeID : ActivationIt->second.Nodes) {
      auto NodeIt = NoAliasNodes.find(NodeID);
      if (NodeIt != NoAliasNodes.end() && NodeIt->second.Object == &MO)
        NodeIt->second.Active = false;
      tryEraseInactiveNoAliasNode(NodeID);
    }
    ActivationIt->second.Accesses.erase(const_cast<MemoryObject *>(&MO));
  }
  NoAliasActivationsByObject.erase(It);
}

} // namespace llvm::ubi
