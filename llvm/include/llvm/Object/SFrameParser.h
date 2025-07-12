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
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace object {

template <endianness E> class SFrameParser {
public:
  static Expected<SFrameParser> create(ArrayRef<uint8_t> Contents);

  const sframe::Preamble<E> &getPreamble() const { return Header.Preamble; }
  const sframe::Header<E> &getHeader() const { return Header; }

  bool usesFixedRAOffset() const {
    return getHeader().ABIArch == sframe::ABI::AMD64EndianLittle;
  }
  bool usesFixedFPOffset() const {
    return false; // Not used in any currently defined ABI.
  }

private:
  ArrayRef<uint8_t> Data;
  const sframe::Header<E> &Header;

  SFrameParser(ArrayRef<uint8_t> Data, const sframe::Header<E> &Header)
      : Data(Data), Header(Header) {}
};

extern template class SFrameParser<endianness::big>;
extern template class SFrameParser<endianness::little>;

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_SFRAME_H
