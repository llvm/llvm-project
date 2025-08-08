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
static Expected<const T &> getDataSliceAs(ArrayRef<uint8_t> Data,
                                          uint64_t Offset) {
  static_assert(std::is_trivial_v<T>);
  Expected<ArrayRef<uint8_t>> Slice = getDataSlice(Data, Offset, sizeof(T));
  if (!Slice)
    return Slice.takeError();

  return *reinterpret_cast<const T *>(Slice->data());
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

template class LLVM_EXPORT_TEMPLATE llvm::object::SFrameParser<endianness::big>;
template class LLVM_EXPORT_TEMPLATE
    llvm::object::SFrameParser<endianness::little>;
