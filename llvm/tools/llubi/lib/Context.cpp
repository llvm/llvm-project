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
                              DL.getProgramAddressSpace(), MemInitKind::Zeroed,
                              MemAllocKind::Global);
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
                               MemInitKind::Zeroed, MemAllocKind::Global);
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
    return Pointer::null(C->getType()->getPointerAddressSpace(), DL);

  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    if (auto *VecTy = dyn_cast<VectorType>(CI->getType()))
      return std::vector<AnyValue>(getEVL(VecTy->getElementCount()),
                                   AnyValue(CI->getValue()));
    return CI->getValue();
  }

  if (auto *CFP = dyn_cast<ConstantFP>(C)) {
    if (auto *VecTy = dyn_cast<VectorType>(CFP->getType()))
      return std::vector<AnyValue>(getEVL(VecTy->getElementCount()),
                                   AnyValue(CFP->getValue()));
    return CFP->getValue();
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

AnyValue Context::fromBytes(ConstBytesView Bytes, Type *Ty,
                            uint32_t OffsetInBits, bool CheckPaddingBits,
                            bool *ContainsUndefinedBits) {
  uint32_t NumBits = DL.getTypeSizeInBits(Ty).getFixedValue();
  uint32_t NewOffsetInBits = OffsetInBits + NumBits;
  if (CheckPaddingBits)
    NewOffsetInBits = alignTo(NewOffsetInBits, 8);
  bool NeedsPadding = NewOffsetInBits != OffsetInBits + NumBits;
  uint32_t NumBitsToExtract = NewOffsetInBits - OffsetInBits;
  SmallVector<uint64_t> RawBits(alignTo(NumBitsToExtract, 8));
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

      if (UndefBehavior == UndefValueBehavior::NonDeterministic) {
        // We don't use std::uniform_int_distribution here because it produces
        // different results across different library implementations. Instead,
        // we directly use the low bits from Rng.
        RandomBits = static_cast<uint8_t>(Rng());
      }
    }
    uint8_t ActualBits = ((LogicalByte.Value & LogicalByte.ConcreteMask) |
                          (RandomBits & ~LogicalByte.ConcreteMask)) &
                         Mask;
    RawBits[I / 64] |= static_cast<APInt::WordType>(ActualBits) << (I % 64);
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
  // TODO: recover provenance
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
  auto WriteBits = [&](const APInt &Bits) {
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
    WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits) : Bits);
  } else if (Ty->isFloatingPointTy()) {
    auto Bits = Val.asFloat().bitcastToAPInt();
    WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits) : Bits);
  } else if (Ty->isPointerTy()) {
    auto &Bits = Val.asPointer().address();
    WriteBits(NeedsPadding ? Bits.zext(NewOffsetInBits - OffsetInBits) : Bits);
    // TODO: save metadata of the pointer.
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

void Context::freeze(AnyValue &Val, Type *Ty) {
  if (Val.isPoison()) {
    uint32_t Bits = DL.getTypeSizeInBits(Ty);
    APInt RandomVal = APInt::getZero(Bits);
    if (UndefBehavior == UndefValueBehavior::NonDeterministic) {
      SmallVector<APInt::WordType> RandomWords;
      uint32_t NumWords = APInt::getNumWords(Bits);
      RandomWords.reserve(NumWords);
      static_assert(decltype(Rng)::word_size >=
                        std::numeric_limits<APInt::WordType>::digits,
                    "Unexpected Rng result type.");
      for (uint32_t I = 0; I != NumWords; ++I)
        RandomWords.push_back(static_cast<APInt::WordType>(Rng()));
      RandomVal = APInt(Bits, RandomWords);
    }
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

MemoryObject::~MemoryObject() = default;
MemoryObject::MemoryObject(uint64_t Addr, uint64_t Size, StringRef Name,
                           unsigned AS, MemInitKind InitKind,
                           MemAllocKind AllocKind)
    : Address(Addr), Size(Size), Name(Name), AS(AS),
      State(InitKind != MemInitKind::Poisoned ? MemoryObjectState::Alive
                                              : MemoryObjectState::Dead),
      AllocKind(AllocKind) {
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
                  MemInitKind InitKind, MemAllocKind AllocKind) {
  // Even if the memory object is zero-sized, it still occupies a byte to obtain
  // a unique address.
  uint64_t AllocateSize = std::max(Size, (uint64_t)1);
  if (MaxMem != 0 && SaturatingAdd(UsedMem, AllocateSize) >= MaxMem)
    return nullptr;
  uint64_t AlignedAddr = alignTo(AllocationBase, Align);
  auto MemObj = makeIntrusiveRefCnt<MemoryObject>(AlignedAddr, Size, Name, AS,
                                                  InitKind, AllocKind);
  MemoryObjects[AlignedAddr] = MemObj;
  AllocationBase = AlignedAddr + AllocateSize;
  UsedMem += AllocateSize;
  return MemObj;
}

bool Context::free(const MemoryObject &Obj) {
  uint64_t Address = Obj.getAddress();
  auto It = MemoryObjects.find(Address);
  if (It == MemoryObjects.end() || It->second.get() != &Obj)
    return false;

  UsedMem -= std::max(It->second->getSize(), static_cast<uint64_t>(1));
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
  // FIXME: It is incorrect for overaligned scalable vector types.
  return getEffectiveTypeSize(DL.getTypeAllocSize(Ty));
}
uint64_t Context::getEffectiveTypeStoreSize(Type *Ty) {
  return getEffectiveTypeSize(DL.getTypeStoreSize(Ty));
}

void MemoryObject::markAsFreed() {
  State = MemoryObjectState::Freed;
  Bytes.clear();
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
