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

using namespace llvm;
using namespace llvm::object;

template <typename T>
static Expected<const T &> getDataSliceAs(ArrayRef<uint8_t> Data,
                                          uint64_t Offset) {
  static_assert(std::is_trivial_v<T>);
  if (Data.size() < Offset + sizeof(T)) {
    return createStringError(
        formatv("unexpected end of data at offset {0:x} while reading [{1:x}, "
                "{2:x})",
                Data.size(), Offset, Offset + sizeof(T))
            .str(),
        object_error::unexpected_eof);
  }
  return *reinterpret_cast<const T *>(Data.data() + Offset);
}

template <endianness E>
Expected<SFrameParser<E>> SFrameParser<E>::create(ArrayRef<uint8_t> Contents) {
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
  return SFrameParser(Contents, *Header);
}

template class llvm::object::SFrameParser<endianness::big>;
template class llvm::object::SFrameParser<endianness::little>;
