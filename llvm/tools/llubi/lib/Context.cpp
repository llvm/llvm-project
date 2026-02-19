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
#include "llvm/Support/MathExtras.h"

namespace llvm::ubi {

Context::Context(Module &M)
    : Ctx(M.getContext()), M(M), DL(M.getDataLayout()),
      TLIImpl(M.getTargetTriple()) {}

Context::~Context() = default;

bool Context::initGlobalValues() {
  // Register all function and block targets that may be used by indirect calls
  // and branches.
  for (Function &F : M) {
    if (F.hasAddressTaken()) {
      // TODO: Use precise alignment for function pointers if it is necessary.
      auto FuncObj = allocate(0, F.getPointerAlignment(DL).value(), F.getName(),
                              DL.getProgramAddressSpace(), MemInitKind::Zeroed);
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
                               MemInitKind::Zeroed);
      if (!BlockObj)
        return false;
      ValidBlockTargets.try_emplace(BlockObj->getAddress(),
                                    std::make_pair(&BB, BlockObj));
      BlockAddrMap.try_emplace(&BB, deriveFromMemoryObject(BlockObj));
    }
  }
  // TODO: initialize global variables.
  return true;
}

AnyValue Context::getConstantValueImpl(Constant *C) {
  if (isa<PoisonValue>(C))
    return AnyValue::getPoisonValue(*this, C->getType());

  if (isa<ConstantAggregateZero>(C))
    return AnyValue::getNullValue(*this, C->getType());

  if (isa<ConstantPointerNull>(C))
    return Pointer::null(
        DL.getPointerSizeInBits(C->getType()->getPointerAddressSpace()));

  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    if (auto *VecTy = dyn_cast<VectorType>(CI->getType()))
      return std::vector<AnyValue>(getEVL(VecTy->getElementCount()),
                                   AnyValue(CI->getValue()));
    return CI->getValue();
  }

  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    std::vector<AnyValue> Elts;
    Elts.reserve(CDS->getNumElements());
    for (uint32_t I = 0, E = CDS->getNumElements(); I != E; ++I)
      Elts.push_back(getConstantValue(CDS->getElementAsConstant(I)));
    return std::move(Elts);
  }

  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    std::vector<AnyValue> Elts;
    Elts.reserve(CA->getNumOperands());
    for (uint32_t I = 0, E = CA->getNumOperands(); I != E; ++I)
      Elts.push_back(getConstantValue(CA->getOperand(I)));
    return std::move(Elts);
  }

  if (auto *BA = dyn_cast<BlockAddress>(C))
    return BlockAddrMap.at(BA->getBasicBlock());

  if (auto *F = dyn_cast<Function>(C))
    return FuncAddrMap.at(F);

  llvm_unreachable("Unrecognized constant");
}

const AnyValue &Context::getConstantValue(Constant *C) {
  auto It = ConstCache.find(C);
  if (It != ConstCache.end())
    return It->second;

  return ConstCache.emplace(C, getConstantValueImpl(C)).first->second;
}

AnyValue Context::fromBytes(ArrayRef<Byte> Bytes, Type *Ty,
                            uint32_t &OffsetInBits, bool CheckPaddingBits) {
  if (Ty->isIntegerTy() || Ty->isFloatingPointTy() || Ty->isPointerTy()) {
    uint32_t NumBits = DL.getTypeSizeInBits(Ty).getFixedValue();
    uint32_t NewOffsetInBits = OffsetInBits + NumBits;
    if (CheckPaddingBits)
      NewOffsetInBits = alignTo(NewOffsetInBits, 8);
    bool NeedsPadding = NewOffsetInBits != OffsetInBits + NumBits;
    uint32_t NumBitsToExtract = NewOffsetInBits - OffsetInBits;
    SmallVector<uint64_t> BitsData(alignTo(NumBitsToExtract, 8));
    for (uint32_t I = 0; I < NumBitsToExtract; I += 8) {
      uint32_t NumBitsInByte = std::min(8U, NumBitsToExtract - I);
      uint32_t BitsStart =
          OffsetInBits +
          (DL.isLittleEndian() ? I : (NumBitsToExtract - NumBitsInByte - I));
      uint32_t BitsEnd = BitsStart + NumBitsInByte - 1;
      Byte LogicalByte;
      if (((BitsStart ^ BitsEnd) & ~7) == 0)
        LogicalByte = Bytes[BitsStart / 8].lshr(BitsStart % 8);
      else
        LogicalByte =
            Bytes[BitsStart / 8].fshr(Bytes[BitsEnd / 8], BitsStart % 8);

      uint32_t Mask = (1U << NumBitsInByte) - 1;
      // If any of the bits in the byte is poison, the whole value is poison.
      if (~LogicalByte.ConcreteMask & ~LogicalByte.Value & Mask)
        return AnyValue::poison();
      uint8_t RandomBits = 0;
      if (UndefBehavior == UndefValueBehavior::NonDeterministic &&
          (~LogicalByte.ConcreteMask & Mask)) {
        // This byte contains undef bits.
        std::uniform_int_distribution<uint8_t> Distrib;
        RandomBits = Distrib(Rng);
      }
      uint8_t ActualBits = ((LogicalByte.Value & LogicalByte.ConcreteMask) |
                            (RandomBits & ~LogicalByte.ConcreteMask)) &
                           Mask;
      BitsData[I / 64] |= static_cast<APInt::WordType>(ActualBits) << (I % 64);
    }
    OffsetInBits = NewOffsetInBits;

    APInt Bits(NumBitsToExtract, BitsData);

    // Padding bits for non-byte-sized scalar types must be zero.
    if (NeedsPadding) {
      if (!Bits.isIntN(NumBits))
        return AnyValue::poison();
      Bits = Bits.trunc(NumBits);
    }

    if (Ty->isIntegerTy())
      return Bits;
    if (Ty->isFloatingPointTy())
      return APFloat(Ty->getFltSemantics(), Bits);
    assert(Ty->isPointerTy() && "Expect a pointer type");
    // TODO: recover provenance
    return Pointer(Bits);
  }

  assert(OffsetInBits % 8 == 0 && "Missing padding bits.");
  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    Type *ElemTy = VecTy->getElementType();
    std::vector<AnyValue> ValVec;
    uint32_t NumElements = getEVL(VecTy->getElementCount());
    ValVec.reserve(NumElements);
    for (uint32_t I = 0; I != NumElements; ++I)
      ValVec.push_back(
          fromBytes(Bytes, ElemTy, OffsetInBits, /*CheckPaddingBits=*/false));
    if (DL.isBigEndian())
      std::reverse(ValVec.begin(), ValVec.end());
    return AnyValue(std::move(ValVec));
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    Type *ElemTy = ArrTy->getElementType();
    std::vector<AnyValue> ValVec;
    uint32_t NumElements = ArrTy->getNumElements();
    ValVec.reserve(NumElements);
    for (uint32_t I = 0; I != NumElements; ++I)
      ValVec.push_back(
          fromBytes(Bytes, ElemTy, OffsetInBits, /*CheckPaddingBits=*/true));
    return AnyValue(std::move(ValVec));
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    auto *Layout = DL.getStructLayout(StructTy);
    uint32_t BaseOffsetInBits = OffsetInBits;
    std::vector<AnyValue> ValVec;
    uint32_t NumElements = StructTy->getNumElements();
    ValVec.reserve(NumElements);
    for (uint32_t I = 0; I != NumElements; ++I) {
      Type *ElemTy = StructTy->getElementType(I);
      TypeSize ElemOffset = Layout->getElementOffset(I);
      OffsetInBits =
          BaseOffsetInBits + (ElemOffset.isScalable()
                                  ? ElemOffset.getKnownMinValue() * VScale
                                  : ElemOffset.getFixedValue()) *
                                 8;
      ValVec.push_back(
          fromBytes(Bytes, ElemTy, OffsetInBits, /*CheckPaddingBits=*/true));
    }
    OffsetInBits =
        BaseOffsetInBits +
        static_cast<uint32_t>(getEffectiveTypeStoreSize(StructTy)) * 8;
    return AnyValue(std::move(ValVec));
  }
  llvm_unreachable("Unsupported first class type.");
}

void Context::toBytes(const AnyValue &Val, Type *Ty, uint32_t &OffsetInBits,
                      MutableArrayRef<Byte> Bytes, bool PaddingBits) {
  if (Val.isPoison() || Ty->isIntegerTy() || Ty->isFloatingPointTy() ||
      Ty->isPointerTy()) {
    uint32_t NumBits = DL.getTypeSizeInBits(Ty).getFixedValue();
    uint32_t NewOffsetInBits = OffsetInBits + NumBits;
    if (PaddingBits)
      NewOffsetInBits = alignTo(NewOffsetInBits, 8);
    bool NeedsPadding = NewOffsetInBits != OffsetInBits + NumBits;
    auto WriteBits = [&](const APInt &Bits) {
      for (uint32_t I = 0, E = Bits.getBitWidth(); I < E; I += 8) {
        uint32_t NumBitsInByte = std::min(8U, E - I);
        uint32_t BitsStart =
            OffsetInBits + (DL.isLittleEndian() ? I : (E - NumBitsInByte - I));
        uint32_t BitsEnd = BitsStart + NumBitsInByte - 1;
        uint8_t BitsVal =
            static_cast<uint8_t>(Bits.extractBitsAsZExtValue(NumBitsInByte, I));

        Bytes[BitsStart / 8].writeBits(
            static_cast<uint8_t>(((1U << NumBitsInByte) - 1)
                                 << (BitsStart % 8)),
            static_cast<uint8_t>(BitsVal << (BitsStart % 8)));
        // Crosses the byte boundary.
        if (((BitsStart ^ BitsEnd) & ~7) != 0)
          Bytes[BitsEnd / 8].writeBits(
              static_cast<uint8_t>((1U << (BitsEnd % 8 + 1)) - 1),
              static_cast<uint8_t>(BitsVal >> (8 - (BitsStart % 8))));
      }
    };
    if (Val.isPoison()) {
      for (uint32_t I = 0, E = NewOffsetInBits - OffsetInBits; I < E;) {
        uint32_t NumBitsInByte = std::min(8 - (OffsetInBits + I) % 8, E - I);
        assert(((OffsetInBits ^ (OffsetInBits + NumBitsInByte - 1)) & ~7) ==
                   0 &&
               "Across byte boundary.");
        Bytes[(OffsetInBits + I) / 8].poisonBits(static_cast<uint8_t>(
            ((1U << NumBitsInByte) - 1) << ((OffsetInBits + I) % 8)));
        I += NumBitsInByte;
      }
    } else if (Ty->isIntegerTy()) {
      auto &Bits = Val.asInteger();
      WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits)
                             : Bits);
    } else if (Ty->isFloatingPointTy()) {
      auto Bits = Val.asFloat().bitcastToAPInt();
      WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits)
                             : Bits);
    } else if (Ty->isPointerTy()) {
      auto &Bits = Val.asPointer().address();
      WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits)
                             : Bits);
      // TODO: save metadata of the pointer.
    } else {
      llvm_unreachable("Unsupported scalar type.");
    }
    OffsetInBits = NewOffsetInBits;
    return;
  }

  assert(OffsetInBits % 8 == 0 && "Missing padding bits.");
  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    Type *ElemTy = VecTy->getElementType();
    auto &ValVec = Val.asAggregate();
    uint32_t NewOffsetInBits =
        alignTo(OffsetInBits + DL.getTypeSizeInBits(ElemTy).getFixedValue() *
                                   ValVec.size(),
                8);
    if (DL.isLittleEndian()) {
      for (const auto &SubVal : ValVec)
        toBytes(SubVal, ElemTy, OffsetInBits, Bytes,
                /*PaddingBits=*/false);
    } else {
      for (const auto &SubVal : reverse(ValVec))
        toBytes(SubVal, ElemTy, OffsetInBits, Bytes,
                /*PaddingBits=*/false);
    }
    if (NewOffsetInBits != OffsetInBits) {
      assert(OffsetInBits % 8 != 0 && NewOffsetInBits - OffsetInBits < 8 &&
             "Unexpected offset.");
      // Fill remaining bits with undef.
      Bytes[OffsetInBits / 8].undefBits(
          static_cast<uint8_t>(~0U << (OffsetInBits % 8)));
    }
    OffsetInBits = NewOffsetInBits;
    return;
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    Type *ElemTy = ArrTy->getElementType();
    for (const auto &SubVal : Val.asAggregate())
      toBytes(SubVal, ElemTy, OffsetInBits, Bytes, /*PaddingBits=*/true);
    return;
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    auto *Layout = DL.getStructLayout(StructTy);
    uint32_t BaseOffsetInBits = OffsetInBits;
    auto FillUndefBytes = [&](uint32_t NewOffsetInBits) {
      if (OffsetInBits == NewOffsetInBits)
        return;
      // Fill padding bits due to alignment requirement.
      assert(NewOffsetInBits > OffsetInBits &&
             "Unexpected negative padding bits!");
      fill(Bytes.slice(OffsetInBits / 8, (NewOffsetInBits - OffsetInBits) / 8),
           Byte::undef());
      OffsetInBits = NewOffsetInBits;
    };
    for (uint32_t I = 0, E = Val.asAggregate().size(); I != E; ++I) {
      Type *ElemTy = StructTy->getElementType(I);
      TypeSize ElemOffset = Layout->getElementOffset(I);
      uint32_t NewOffsetInBits =
          BaseOffsetInBits + (ElemOffset.isScalable()
                                  ? ElemOffset.getKnownMinValue() * VScale
                                  : ElemOffset.getFixedValue()) *
                                 8;
      FillUndefBytes(NewOffsetInBits);
      toBytes(Val.asAggregate()[I], ElemTy, OffsetInBits, Bytes,
              /*PaddingBits=*/true);
    }
    uint32_t NewOffsetInBits =
        BaseOffsetInBits + getEffectiveTypeStoreSize(StructTy) * 8;
    FillUndefBytes(NewOffsetInBits);
    return;
  }

  llvm_unreachable("Unsupported first class type.");
}

AnyValue Context::fromBytes(ArrayRef<Byte> Bytes, Type *Ty) {
  uint32_t OffsetInBits = 0;
  return fromBytes(Bytes, Ty, OffsetInBits, /*CheckPaddingBits=*/true);
}

void Context::toBytes(const AnyValue &Val, Type *Ty,
                      MutableArrayRef<Byte> Bytes) {
  uint32_t OffsetInBits = 0;
  toBytes(Val, Ty, OffsetInBits, Bytes, /*PaddingBits=*/true);
}

AnyValue Context::load(MemoryObject &MO, uint64_t Offset, Type *ValTy) {
  return fromBytes(
      MO.getBytes().slice(Offset, getEffectiveTypeStoreSize(ValTy)), ValTy);
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

MemoryObject::~MemoryObject() = default;
MemoryObject::MemoryObject(uint64_t Addr, uint64_t Size, StringRef Name,
                           unsigned AS, MemInitKind InitKind)
    : Address(Addr), Size(Size), Name(Name), AS(AS),
      State(InitKind != MemInitKind::Poisoned ? MemoryObjectState::Alive
                                              : MemoryObjectState::Dead) {
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

IntrusiveRefCntPtr<MemoryObject> Context::allocate(uint64_t Size,
                                                   uint64_t Align,
                                                   StringRef Name, unsigned AS,
                                                   MemInitKind InitKind) {
  // Even if the memory object is zero-sized, it still occupies a byte to obtain
  // a unique address.
  uint64_t AllocateSize = std::max(Size, (uint64_t)1);
  if (MaxMem != 0 && SaturatingAdd(UsedMem, AllocateSize) >= MaxMem)
    return nullptr;
  uint64_t AlignedAddr = alignTo(AllocationBase, Align);
  auto MemObj =
      makeIntrusiveRefCnt<MemoryObject>(AlignedAddr, Size, Name, AS, InitKind);
  MemoryObjects[AlignedAddr] = MemObj;
  AllocationBase = AlignedAddr + AllocateSize;
  UsedMem += AllocateSize;
  return MemObj;
}

bool Context::free(uint64_t Address) {
  auto It = MemoryObjects.find(Address);
  if (It == MemoryObjects.end())
    return false;
  UsedMem -= std::max(It->second->getSize(), (uint64_t)1);
  It->second->markAsFreed();
  MemoryObjects.erase(It);
  return true;
}

Pointer Context::deriveFromMemoryObject(IntrusiveRefCntPtr<MemoryObject> Obj) {
  assert(Obj && "Cannot determine the address space of a null memory object");
  return Pointer(Obj, APInt(DL.getPointerSizeInBits(Obj->getAddressSpace()),
                            Obj->getAddress()));
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
  TypeSize Size = DL.getTypeAllocSize(Ty);
  if (Size.isScalable())
    return Size.getKnownMinValue() * VScale;
  return Size.getFixedValue();
}
uint64_t Context::getEffectiveTypeStoreSize(Type *Ty) {
  TypeSize Size = DL.getTypeStoreSize(Ty);
  if (Size.isScalable())
    return Size.getKnownMinValue() * VScale;
  return Size.getFixedValue();
}

void MemoryObject::markAsFreed() {
  State = MemoryObjectState::Freed;
  Bytes.clear();
}

} // namespace llvm::ubi
