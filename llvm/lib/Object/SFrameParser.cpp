//===- SFrameParser.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SFrameParser.h"
#include "llvm/BinaryFormat/SFrame.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::object;

static Expected<ArrayRef<uint8_t>>
getDataSlice(ArrayRef<uint8_t> Data, uint64_t Offset, uint64_t Size) {
  uint64_t End = SaturatingAdd(Offset, Size);
  // Data.size() cannot be UINT64_MAX, as it would occupy the whole address
  // space.
  if (End > Data.size()) {
    return createStringError(
        formatv("unexpected end of data at offset {0:x} while reading [{1:x}, "
                "{2:x})",
                Data.size(), Offset, End)
            .str(),
        object_error::unexpected_eof);
  }
  return Data.slice(Offset, Size);
}

template <typename T>
static Expected<ArrayRef<T>>
getDataSliceAsArrayOf(ArrayRef<uint8_t> Data, uint64_t Offset, uint64_t Count) {
  static_assert(std::is_trivial_v<T>);
  Expected<ArrayRef<uint8_t>> Slice =
      getDataSlice(Data, Offset, sizeof(T) * Count);
  if (!Slice)
    return Slice.takeError();

  return ArrayRef(reinterpret_cast<const T *>(Slice->data()), Count);
}

template <typename T>
static Expected<const T &> getDataSliceAs(ArrayRef<uint8_t> Data,
                                          uint64_t Offset) {
  Expected<ArrayRef<T>> Array = getDataSliceAsArrayOf<T>(Data, Offset, 1);
  if (!Array)
    return Array.takeError();

  return Array->front();
}

template <endianness E>
Expected<SFrameParser<E>> SFrameParser<E>::create(ArrayRef<uint8_t> Contents,
                                                  uint64_t SectionAddress) {
  Expected<const sframe::Preamble<E> &> Preamble =
      getDataSliceAs<sframe::Preamble<E>>(Contents, 0);
  if (!Preamble)
    return Preamble.takeError();

  if (Preamble->Magic != sframe::Magic)
    return createError(
        formatv("invalid magic number ({0:x+4})", Preamble->Magic.value()));
  if (Preamble->Version != sframe::Version::V2)
    return createError(
        formatv("invalid/unsupported version number ({0})",
                static_cast<unsigned>(Preamble->Version.value())));

  Expected<const sframe::Header<E> &> Header =
      getDataSliceAs<sframe::Header<E>>(Contents, 0);
  if (!Header)
    return Header.takeError();
  return SFrameParser(Contents, SectionAddress, *Header);
}

template <endianness E>
Expected<ArrayRef<uint8_t>> SFrameParser<E>::getAuxHeader() const {
  return getDataSlice(Data, sizeof(Header), Header.AuxHdrLen);
}

template <endianness E>
Expected<ArrayRef<sframe::FuncDescEntry<E>>> SFrameParser<E>::fdes() const {
  Expected<ArrayRef<uint8_t>> Slice = getDataSlice(
      Data, getFDEBase(), Header.NumFDEs * sizeof(sframe::FuncDescEntry<E>));
  if (!Slice)
    return Slice.takeError();
  return ArrayRef(
      reinterpret_cast<const sframe::FuncDescEntry<E> *>(Slice->data()),
      Header.NumFDEs);
}

template <endianness E>
uint64_t SFrameParser<E>::getAbsoluteStartAddress(
    typename FDERange::iterator FDE) const {
  uint64_t Result = SectionAddress + FDE->StartAddress;

  if ((getPreamble().Flags.value() & sframe::Flags::FDEFuncStartPCRel) ==
      sframe::Flags::FDEFuncStartPCRel) {
    uintptr_t DataPtr = reinterpret_cast<uintptr_t>(Data.data());
    uintptr_t FDEPtr = reinterpret_cast<uintptr_t>(&*FDE);

    assert(DataPtr <= FDEPtr && FDEPtr < DataPtr + Data.size() &&
           "Iterator does not belong to this object!");

    Result += FDEPtr - DataPtr;
  }

  return Result;
}

template <typename EndianT>
static Error readArray(ArrayRef<uint8_t> Data, uint64_t Count, uint64_t &Offset,
                       SmallVectorImpl<int32_t> &Vec) {
  Expected<ArrayRef<EndianT>> RawArray =
      getDataSliceAsArrayOf<EndianT>(Data, Offset, Count);
  if (!RawArray)
    return RawArray.takeError();
  Offset += Count * sizeof(EndianT);
  Vec.resize(Count);
  llvm::copy(*RawArray, Vec.begin());
  return Error::success();
}

template <typename T, endianness E>
static Error readFRE(ArrayRef<uint8_t> Data, uint64_t &Offset,
                     typename SFrameParser<E>::FrameRowEntry &FRE) {
  Expected<sframe::FrameRowEntry<T, E>> RawFRE =
      getDataSliceAs<sframe::FrameRowEntry<T, E>>(Data, Offset);
  if (!RawFRE)
    return RawFRE.takeError();

  Offset += sizeof(*RawFRE);
  FRE.StartAddress = RawFRE->StartAddress;
  FRE.Info.Info = RawFRE->Info.Info;

  switch (FRE.Info.getOffsetSize()) {
  case sframe::FREOffset::B1:
    return readArray<sframe::detail::packed<int8_t, E>>(
        Data, FRE.Info.getOffsetCount(), Offset, FRE.Offsets);
  case sframe::FREOffset::B2:
    return readArray<sframe::detail::packed<int16_t, E>>(
        Data, FRE.Info.getOffsetCount(), Offset, FRE.Offsets);
  case sframe::FREOffset::B4:
    return readArray<sframe::detail::packed<int32_t, E>>(
        Data, FRE.Info.getOffsetCount(), Offset, FRE.Offsets);
  default:
    return createError("unsupported/unknown offset size");
  }
}

template <endianness E> Error SFrameParser<E>::FallibleFREIterator::inc() {
  if (++Idx == Size)
    return Error::success();

  switch (FREType) {
  case sframe::FREType::Addr1:
    return readFRE<uint8_t, E>(Data, Offset, FRE);
  case sframe::FREType::Addr2:
    return readFRE<uint16_t, E>(Data, Offset, FRE);
  case sframe::FREType::Addr4:
    return readFRE<uint32_t, E>(Data, Offset, FRE);
  default:
    return createError("invalid/unsupported FRE type");
  }
}

template <endianness E>
iterator_range<typename SFrameParser<E>::fre_iterator>
SFrameParser<E>::fres(const sframe::FuncDescEntry<E> &FDE, Error &Err) const {
  uint64_t Offset = getFREBase() + FDE.StartFREOff;
  fre_iterator BeforeBegin = make_fallible_itr(
      FallibleFREIterator(Data, FDE.getFREType(), -1, FDE.NumFREs, Offset),
      Err);
  fre_iterator End = make_fallible_end(
      FallibleFREIterator(Data, FDE.getFREType(), FDE.NumFREs, FDE.NumFREs,
                          /*Offset=*/0));
  return {++BeforeBegin, End};
}

static std::optional<int32_t> getOffset(ArrayRef<int32_t> Offsets, size_t Idx) {
  if (Offsets.size() > Idx)
    return Offsets[Idx];
  return std::nullopt;
}

// The interpretation of offsets is ABI-specific. The implementation of this and
// the following functions may need to be adjusted when adding support for a new
// ABI.
template <endianness E>
std::optional<int32_t>
SFrameParser<E>::getCFAOffset(const FrameRowEntry &FRE) const {
  return getOffset(FRE.Offsets, 0);
}

template <endianness E>
std::optional<int32_t>
SFrameParser<E>::getRAOffset(const FrameRowEntry &FRE) const {
  if (usesFixedRAOffset())
    return Header.CFAFixedRAOffset;
  return getOffset(FRE.Offsets, 1);
}

template <endianness E>
std::optional<int32_t>
SFrameParser<E>::getFPOffset(const FrameRowEntry &FRE) const {
  if (usesFixedFPOffset())
    return Header.CFAFixedFPOffset;
  return getOffset(FRE.Offsets, usesFixedRAOffset() ? 1 : 2);
}

template <endianness E>
ArrayRef<int32_t>
SFrameParser<E>::getExtraOffsets(const FrameRowEntry &FRE) const {
  size_t UsedOffsets = 1; // CFA
  if (!usesFixedRAOffset())
    ++UsedOffsets;
  if (!usesFixedFPOffset())
    ++UsedOffsets;
  if (FRE.Offsets.size() > UsedOffsets)
    return ArrayRef(FRE.Offsets).drop_front(UsedOffsets);
  return {};
}

template class LLVM_EXPORT_TEMPLATE llvm::object::SFrameParser<endianness::big>;
template class LLVM_EXPORT_TEMPLATE
    llvm::object::SFrameParser<endianness::little>;
