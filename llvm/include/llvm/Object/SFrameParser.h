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
#include "llvm/BinaryFormat/SFrame.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace object {

template <endianness E> class SFrameParser {
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
};

extern template class LLVM_TEMPLATE_ABI SFrameParser<endianness::big>;
extern template class LLVM_TEMPLATE_ABI SFrameParser<endianness::little>;

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_SFRAME_H
