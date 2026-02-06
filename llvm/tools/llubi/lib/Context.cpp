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

AnyValue Context::getConstantValueImpl(Constant *C) {
  if (isa<PoisonValue>(C))
    return AnyValue::getPoisonValue(*this, C->getType());

  // TODO: Handle ConstantInt vector.
  if (auto *CI = dyn_cast<ConstantInt>(C))
    return CI->getValue();

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
  default:
    llvm_unreachable("Unknown MemInitKind");
  }
}

IntrusiveRefCntPtr<MemoryObject> Context::allocate(uint64_t Size,
                                                   uint64_t Align,
                                                   StringRef Name, unsigned AS,
                                                   MemInitKind InitKind) {
  if (MaxMem != 0 && SaturatingAdd(UsedMem, Size) >= MaxMem)
    return nullptr;
  uint64_t AlignedAddr = alignTo(AllocationBase, Align);
  auto MemObj =
      makeIntrusiveRefCnt<MemoryObject>(AlignedAddr, Size, Name, AS, InitKind);
  MemoryObjects[AlignedAddr] = MemObj;
  AllocationBase = AlignedAddr + Size;
  UsedMem += Size;
  return MemObj;
}

bool Context::free(uint64_t Address) {
  auto It = MemoryObjects.find(Address);
  if (It == MemoryObjects.end())
    return false;
  UsedMem -= It->second->getSize();
  It->second->markAsFreed();
  MemoryObjects.erase(It);
  return true;
}

Pointer Context::deriveFromMemoryObject(IntrusiveRefCntPtr<MemoryObject> Obj) {
  assert(Obj && "Cannot determine the address space of a null memory object");
  return Pointer(
      Obj,
      APInt(DL.getPointerSizeInBits(Obj->getAddressSpace()), Obj->getAddress()),
      /*Offset=*/0);
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
