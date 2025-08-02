//===- SFrameParser.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_SFRAME_H
#define LLVM_OBJECT_SFRAME_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/fallible_iterator.h"
#include "llvm/BinaryFormat/SFrame.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace object {

template <endianness E> class SFrameParser {
  class FallibleFREIterator;

public:
  static Expected<SFrameParser> create(ArrayRef<uint8_t> Contents,
                                       uint64_t SectionAddress);

  const sframe::Preamble<E> &getPreamble() const { return Header.Preamble; }
  const sframe::Header<E> &getHeader() const { return Header; }

  Expected<ArrayRef<uint8_t>> getAuxHeader() const;

  bool usesFixedRAOffset() const {
    return getHeader().ABIArch == sframe::ABI::AMD64EndianLittle;
  }
  bool usesFixedFPOffset() const {
    return false; // Not used in any currently defined ABI.
  }

  using FDERange = ArrayRef<sframe::FuncDescEntry<E>>;
  Expected<FDERange> fdes() const;

  // Decodes the start address of the given FDE, which must be one of the
  // objects returned by the `fdes()` function.
  uint64_t getAbsoluteStartAddress(typename FDERange::iterator FDE) const;

  struct FrameRowEntry {
    uint32_t StartAddress;
    sframe::FREInfo<endianness::native> Info;
    SmallVector<int32_t, 3> Offsets;
  };

  using fre_iterator = fallible_iterator<FallibleFREIterator>;
  iterator_range<fre_iterator> fres(const sframe::FuncDescEntry<E> &FDE,
                                    Error &Err) const;

  std::optional<int32_t> getCFAOffset(const FrameRowEntry &FRE) const;
  std::optional<int32_t> getRAOffset(const FrameRowEntry &FRE) const;
  std::optional<int32_t> getFPOffset(const FrameRowEntry &FRE) const;
  ArrayRef<int32_t> getExtraOffsets(const FrameRowEntry &FRE) const;

private:
  ArrayRef<uint8_t> Data;
  uint64_t SectionAddress;
  const sframe::Header<E> &Header;

  SFrameParser(ArrayRef<uint8_t> Data, uint64_t SectionAddress,
               const sframe::Header<E> &Header)
      : Data(Data), SectionAddress(SectionAddress), Header(Header) {}

  uint64_t getFDEBase() const {
    return sizeof(Header) + Header.AuxHdrLen + Header.FDEOff;
  }

  uint64_t getFREBase() const {
    return getFDEBase() + Header.NumFDEs * sizeof(sframe::FuncDescEntry<E>);
  }
};

template <endianness E> class SFrameParser<E>::FallibleFREIterator {
public:
  // NB: This iterator starts out in the before_begin() state. It must be
  // ++'ed to reach the first element.
  FallibleFREIterator(ArrayRef<uint8_t> Data, sframe::FREType FREType,
                      uint32_t Idx, uint32_t Size, uint64_t Offset)
      : Data(Data), FREType(FREType), Idx(Idx), Size(Size), Offset(Offset) {}

  Error inc();
  const FrameRowEntry &operator*() const { return FRE; }

  friend bool operator==(const FallibleFREIterator &LHS,
                         const FallibleFREIterator &RHS) {
    assert(LHS.Data.data() == RHS.Data.data());
    assert(LHS.Data.size() == RHS.Data.size());
    assert(LHS.FREType == RHS.FREType);
    assert(LHS.Size == RHS.Size);
    return LHS.Idx == RHS.Idx;
  }

private:
  ArrayRef<uint8_t> Data;
  sframe::FREType FREType;
  uint32_t Idx;
  uint32_t Size;
  uint64_t Offset;
  FrameRowEntry FRE;
};

extern template class LLVM_TEMPLATE_ABI SFrameParser<endianness::big>;
extern template class LLVM_TEMPLATE_ABI SFrameParser<endianness::little>;

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_SFRAME_H
