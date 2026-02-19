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

MemoryObject::~MemoryObject() = default;
MemoryObject::MemoryObject(uint64_t Addr, uint64_t Size, StringRef Name,
                           unsigned AS, MemInitKind InitKind)
    : Address(Addr), Size(Size), Name(Name), AS(AS),
      State(InitKind != MemInitKind::Poisoned ? MemoryObjectState::Alive
                                              : MemoryObjectState::Dead) {
  switch (InitKind) {
  case MemInitKind::Zeroed:
    Bytes.resize(Size, Byte{0, ByteKind::Concrete});
    break;
  case MemInitKind::Uninitialized:
    Bytes.resize(Size, Byte{0, ByteKind::Undef});
    break;
  case MemInitKind::Poisoned:
    Bytes.resize(Size, Byte{0, ByteKind::Poison});
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

void MemoryObject::markAsFreed() {
  State = MemoryObjectState::Freed;
  Bytes.clear();
}

void MemoryObject::writeRawBytes(uint64_t Offset, const void *Data,
                                 uint64_t Length) {
  assert(SaturatingAdd(Offset, Length) <= Size && "Write out of bounds");
  const uint8_t *ByteData = static_cast<const uint8_t *>(Data);
  for (uint64_t I = 0; I < Length; ++I)
    Bytes[Offset + I].set(ByteData[I]);
}

void MemoryObject::writeInteger(uint64_t Offset, const APInt &Int,
                                const DataLayout &DL) {
  uint64_t BitWidth = Int.getBitWidth();
  uint64_t IntSize = divideCeil(BitWidth, 8);
  assert(SaturatingAdd(Offset, IntSize) <= Size && "Write out of bounds");
  for (uint64_t I = 0; I < IntSize; ++I) {
    uint64_t ByteIndex = DL.isLittleEndian() ? I : (IntSize - 1 - I);
    uint64_t Bits = std::min(BitWidth - ByteIndex * 8, uint64_t(8));
    Bytes[Offset + I].set(Int.extractBitsAsZExtValue(Bits, ByteIndex * 8));
  }
}
void MemoryObject::writeFloat(uint64_t Offset, const APFloat &Float,
                              const DataLayout &DL) {
  writeInteger(Offset, Float.bitcastToAPInt(), DL);
}
void MemoryObject::writePointer(uint64_t Offset, const Pointer &Ptr,
                                const DataLayout &DL) {
  writeInteger(Offset, Ptr.address(), DL);
  // TODO: provenance
}

} // namespace llvm::ubi
