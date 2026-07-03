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
  case Instruction::PtrToAddr: {
    const auto *Src = getConstantValue(CE->getOperand(0));
    if (!Src)
      return std::nullopt;
    if (Src->isPoison())
      return MaterializedConstant(AnyValue::poison(), Src->isCacheable());
    unsigned BitWidth = CE->getType()->getScalarSizeInBits();
    if (Src->isPointer())
      return MaterializedConstant(Src->asPointer().address().trunc(BitWidth),
                                  Src->isCacheable());
    std::vector<AnyValue> Vec = Src->asAggregate();
    for (auto &V : Vec) {
      if (V.isPointer())
        V = V.asPointer().address().trunc(BitWidth);
    }
    return MaterializedConstant(std::move(Vec), Src->isCacheable());
  }
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
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
      return Pointer(std::move(Prov), Bits);
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
  auto WriteBits = [&](const APInt &Bits, const APInt *TagBits) {
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
              /*TagBits=*/nullptr);
  } else if (Ty->isFloatingPointTy()) {
    auto Bits = Val.asFloat().bitcastToAPInt();
    WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits) : Bits,
              /*TagBits=*/nullptr);
  } else if (Ty->isPointerTy()) {
    auto &AddressBits = Val.asPointer().address();
    APInt Tag = getTag(AddressBits.getBitWidth(), Val.asPointer().provenance());
    if (NeedsPadding)
      Tag = Tag.zext(NewOffsetInBits - OffsetInBits);
    WriteBits(NeedsPadding ? AddressBits.zext(NewOffsetInBits - OffsetInBits)
                           : AddressBits,
              &Tag);
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

} // namespace llvm::ubi
