//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/BTF/BTFBuilder.h"
#include "llvm/DebugInfo/BTF/BTFParser.h"
#include "llvm/Support/SwapByteOrder.h"

using namespace llvm;

BTFBuilder::BTFBuilder() {
  // String table starts with the empty string at offset 0.
  Strings.push_back('\0');
}

uint32_t BTFBuilder::addString(StringRef S) {
  uint32_t Offset = Strings.size();
  Strings.append(S.begin(), S.end());
  Strings.push_back('\0');
  return Offset;
}

uint32_t BTFBuilder::addType(const BTF::CommonType &Header) {
  TypeOffsets.push_back(TypeData.size());
  const auto *Ptr = reinterpret_cast<const uint8_t *>(&Header);
  TypeData.append(Ptr, Ptr + sizeof(Header));
  return TypeOffsets.size(); // 1-based ID
}

const BTF::CommonType *BTFBuilder::findType(uint32_t Id) const {
  if (Id == 0 || Id > TypeOffsets.size())
    return nullptr;
  return reinterpret_cast<const BTF::CommonType *>(
      &TypeData[TypeOffsets[Id - 1]]);
}

// Returns {Start, Size} for a type's byte range, or {0, 0} for invalid IDs.
static std::pair<uint32_t, uint32_t>
typeBounds(uint32_t Id, const SmallVectorImpl<uint32_t> &TypeOffsets,
           size_t TypeDataSize) {
  if (Id == 0 || Id > TypeOffsets.size())
    return {0, 0};
  uint32_t Start = TypeOffsets[Id - 1];
  uint32_t End = (Id < TypeOffsets.size()) ? TypeOffsets[Id] : TypeDataSize;
  return {Start, End - Start};
}

ArrayRef<uint8_t> BTFBuilder::getTypeBytes(uint32_t Id) const {
  auto [Start, Size] = typeBounds(Id, TypeOffsets, TypeData.size());
  if (Size == 0)
    return {};
  return ArrayRef<uint8_t>(&TypeData[Start], Size);
}

MutableArrayRef<uint8_t> BTFBuilder::getMutableTypeBytes(uint32_t Id) {
  auto [Start, Size] = typeBounds(Id, TypeOffsets, TypeData.size());
  if (Size == 0)
    return {};
  return MutableArrayRef<uint8_t>(&TypeData[Start], Size);
}

StringRef BTFBuilder::findString(uint32_t Offset) const {
  if (Offset >= Strings.size())
    return StringRef();
  return StringRef(&Strings[Offset]);
}

bool BTFBuilder::hasTypeRef(uint32_t Kind) {
  switch (Kind) {
  case BTF::BTF_KIND_PTR:
  case BTF::BTF_KIND_TYPEDEF:
  case BTF::BTF_KIND_VOLATILE:
  case BTF::BTF_KIND_CONST:
  case BTF::BTF_KIND_RESTRICT:
  case BTF::BTF_KIND_FUNC:
  case BTF::BTF_KIND_FUNC_PROTO:
  case BTF::BTF_KIND_VAR:
  case BTF::BTF_KIND_DECL_TAG:
  case BTF::BTF_KIND_TYPE_TAG:
    return true;
  default:
    return false;
  }
}

// Remap type IDs in a type entry. Non-zero IDs are adjusted by IdDelta.
static void remapTypeIds(uint8_t *Data, uint32_t IdDelta) {
  if (IdDelta == 0)
    return;

  auto *T = reinterpret_cast<BTF::CommonType *>(Data);
  uint8_t *TailPtr = Data + sizeof(BTF::CommonType);

  if (BTFBuilder::hasTypeRef(T->getKind()) && T->Type != 0)
    T->Type += IdDelta;

  switch (T->getKind()) {
  case BTF::BTF_KIND_ARRAY: {
    auto *A = reinterpret_cast<BTF::BTFArray *>(TailPtr);
    if (A->ElemType != 0)
      A->ElemType += IdDelta;
    if (A->IndexType != 0)
      A->IndexType += IdDelta;
    break;
  }
  case BTF::BTF_KIND_STRUCT:
  case BTF::BTF_KIND_UNION: {
    auto *M = reinterpret_cast<BTF::BTFMember *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      if (M[I].Type != 0)
        M[I].Type += IdDelta;
    break;
  }
  case BTF::BTF_KIND_FUNC_PROTO: {
    auto *P = reinterpret_cast<BTF::BTFParam *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      if (P[I].Type != 0)
        P[I].Type += IdDelta;
    break;
  }
  case BTF::BTF_KIND_DATASEC: {
    auto *D = reinterpret_cast<BTF::BTFDataSec *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      if (D[I].Type != 0)
        D[I].Type += IdDelta;
    break;
  }
  default:
    break;
  }
}

// Remap string offsets in a type entry.
static void remapStringOffsets(uint8_t *Data, uint32_t StrDelta) {
  if (StrDelta == 0)
    return;

  auto *T = reinterpret_cast<BTF::CommonType *>(Data);
  T->NameOff += StrDelta;

  uint8_t *TailPtr = Data + sizeof(BTF::CommonType);
  switch (T->getKind()) {
  case BTF::BTF_KIND_STRUCT:
  case BTF::BTF_KIND_UNION: {
    auto *M = reinterpret_cast<BTF::BTFMember *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      M[I].NameOff += StrDelta;
    break;
  }
  case BTF::BTF_KIND_ENUM: {
    auto *E = reinterpret_cast<BTF::BTFEnum *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      E[I].NameOff += StrDelta;
    break;
  }
  case BTF::BTF_KIND_ENUM64: {
    auto *E = reinterpret_cast<BTF::BTFEnum64 *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      E[I].NameOff += StrDelta;
    break;
  }
  case BTF::BTF_KIND_FUNC_PROTO: {
    auto *P = reinterpret_cast<BTF::BTFParam *>(TailPtr);
    for (unsigned I = 0, N = T->getVlen(); I < N; ++I)
      P[I].NameOff += StrDelta;
    break;
  }
  default:
    break;
  }
}

Expected<uint32_t> BTFBuilder::merge(StringRef RawBTFSection,
                                     bool IsLittleEndian) {
  BTFParser Parser;
  if (Error E = Parser.parse(RawBTFSection, IsLittleEndian))
    return std::move(E);

  uint32_t StrDelta = Strings.size();
  uint32_t IdDelta = TypeOffsets.size();
  uint32_t FirstNewId = IdDelta + 1;

  StringRef InputStrings = Parser.getStringTable();
  Strings.append(InputStrings.begin(), InputStrings.end());
  for (uint32_t Id = 1, N = Parser.typesCount(); Id < N; ++Id) {
    uint32_t Offset = TypeData.size();
    ArrayRef<uint8_t> Bytes = Parser.getTypeBytes(Id);
    TypeOffsets.push_back(Offset);
    TypeData.append(Bytes.begin(), Bytes.end());
    remapStringOffsets(&TypeData[Offset], StrDelta);
    remapTypeIds(&TypeData[Offset], IdDelta);
  }

  return FirstNewId;
}

void BTFBuilder::write(SmallVectorImpl<uint8_t> &Out,
                       bool IsLittleEndian) const {
  bool NeedSwap = (IsLittleEndian != sys::IsLittleEndianHost);

  BTF::Header Hdr;
  Hdr.Magic = BTF::MAGIC;
  Hdr.Version = BTF::VERSION;
  Hdr.Flags = 0;
  Hdr.HdrLen = sizeof(BTF::Header);
  Hdr.TypeOff = 0;
  Hdr.TypeLen = TypeData.size();
  Hdr.StrOff = TypeData.size();
  Hdr.StrLen = Strings.size();

  size_t TotalSize = sizeof(Hdr) + TypeData.size() + Strings.size();
  size_t OutStart = Out.size();
  Out.resize(OutStart + TotalSize);
  uint8_t *Buf = &Out[OutStart];

  BTF::Header OutHdr = Hdr;
  if (NeedSwap) {
    sys::swapByteOrder(OutHdr.Magic);
    sys::swapByteOrder(OutHdr.HdrLen);
    sys::swapByteOrder(OutHdr.TypeOff);
    sys::swapByteOrder(OutHdr.TypeLen);
    sys::swapByteOrder(OutHdr.StrOff);
    sys::swapByteOrder(OutHdr.StrLen);
  }
  memcpy(Buf, &OutHdr, sizeof(OutHdr));
  Buf += sizeof(OutHdr);

  memcpy(Buf, TypeData.data(), TypeData.size());
  if (NeedSwap) {
    // Every type record consists of uint32_t fields, regardless of its kind.
    // Use unaligned loads/stores: Out may already contain an arbitrary prefix.
    for (size_t Offset = 0; Offset < TypeData.size();
         Offset += sizeof(uint32_t)) {
      uint32_t Word;
      memcpy(&Word, Buf + Offset, sizeof(Word));
      sys::swapByteOrder(Word);
      memcpy(Buf + Offset, &Word, sizeof(Word));
    }
  }
  Buf += TypeData.size();

  memcpy(Buf, Strings.data(), Strings.size());
}
